import os
import glob
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

from tsl import logger
from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes import DatetimeDataset


TUM_BASE = 'https://dataserv.ub.tum.de/s/m1524895/download'


class WeatherBench1(DatetimeDataset):
    r"""Original WeatherBench (Rasp et al. 2020) benchmark dataset -
    ERA5 reanalysis data regridded to a coarse global grid, pre-packaged
    for ML benchmarking.

    Downloads and caches a single variable's NetCDF data from the
    official TUM data server (https://dataserv.ub.tum.de/s/m1524895).

    Args:
        root (str): Root folder for downloading/caching raw files.
        variable (str): Variable name matching WeatherBench's folder
            naming, e.g. 'geopotential_500', 'temperature_850',
            '2m_temperature', '10m_u_component_of_wind'.
            (default: 'geopotential_500')
        resolution (str): One of '5.625deg' (32x64 grid, ~600km),
            '2.8125deg' (64x128, ~300km), '1.40625deg' (128x256, ~150km).
            (default: '5.625deg')
        start_time (str, optional): ISO date to start the time slice.
        end_time (str, optional): ISO date to end the time slice.
        freq (str, optional): Resampling frequency. Native data is
            hourly for most variables. (default: None, keeps native freq)
    """

    similarity_options = {'distance'}

    def __init__(self,
                 root='./data/weatherbench1',
                 variable='geopotential_500',
                 resolution='5.625deg',
                 start_time=None,
                 end_time=None,
                 freq=None):
        self.root = root
        self.variable = variable
        self.resolution = resolution
        self.start_time = start_time
        self.end_time = end_time

        df, mask, lat_lon = self.load(freq=freq)

        super().__init__(target=df,
                          mask=mask,
                          freq=freq,
                          similarity_score='distance',
                          temporal_aggregation='mean',
                          spatial_aggregation='mean',
                          name='WeatherBench1')

        self.add_covariate('lat_lon', lat_lon, pattern='n c')

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.resolution, self.variable)

    @property
    def zip_filename(self):
        return f'{self.variable}_{self.resolution}.zip'

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        zip_path = os.path.join(self.raw_dir, self.zip_filename)

        nc_files_already_present = glob.glob(
            os.path.join(self.raw_dir, '*.nc'))
        if nc_files_already_present:
            logger.info(f"Found {len(nc_files_already_present)} existing "
                        f".nc files, skipping download.")
            return

        # Confirmed URL pattern from the official pangeo-data/WeatherBench
        # GitHub README - scoped to a single variable's folder.
        url = (f"{TUM_BASE}?path=/{self.resolution}/{self.variable}"
               f"&files={self.zip_filename}")

        logger.info(f"Downloading: {url}")
        logger.info("NOTE: this can be several hundred MB to a few GB "
                     "depending on variable/resolution - not instant.")
        urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Unzipping {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.raw_dir)

    def load_raw(self):
        self.download()
        nc_files = sorted(glob.glob(os.path.join(self.raw_dir, '*.nc')))
        if not nc_files:
            raise FileNotFoundError(
                f"No .nc files found in {self.raw_dir} after download/unzip. "
                f"Check that variable='{self.variable}' and "
                f"resolution='{self.resolution}' are valid - see "
                f"https://github.com/pangeo-data/WeatherBench for the "
                f"exact folder/variable naming."
            )
        logger.info(f"Found {len(nc_files)} .nc files, opening as multi-file dataset")
        ds = xr.open_mfdataset(nc_files, combine='by_coords')
        return ds

    def load(self, freq=None):
        ds = self.load_raw()

        var_names = list(ds.data_vars)
        if len(var_names) == 1:
            da = ds[var_names[0]]
        elif self.variable in ds.data_vars:
            da = ds[self.variable]
        else:
            raise ValueError(
                f"Could not find variable '{self.variable}' in loaded "
                f"dataset. Available: {var_names}"
            )

        if self.start_time is not None or self.end_time is not None:
            da = da.sel(time=slice(self.start_time, self.end_time))

        # WeatherBench1's 5.625deg grid is already small (32x64=2048 nodes)
        # so no coarsening needed by default, unlike WB2's 0.25deg grid.

        lat = da['lat'].values
        lon = da['lon'].values
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        lat_lon = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

        logger.info("Loading data into memory...")
        values = da.values
        values = values.reshape(values.shape[0], -1)

        time_index = pd.to_datetime(da['time'].values)
        df = pd.DataFrame(values, index=time_index)

        if freq is not None:
            df = df.resample(freq).mean()

        mask = ~np.isnan(df.values)
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df, mask, lat_lon

    def compute_similarity(self, method: str, **kwargs):
        if method == 'distance':
            lat_lon = self.lat_lon
            from scipy.spatial.distance import cdist
            dist = cdist(lat_lon, lat_lon, metric='euclidean')
            theta = np.std(dist)
            return gaussian_kernel(dist, theta=theta)
        raise NotImplementedError(f"Unknown similarity method '{method}'")