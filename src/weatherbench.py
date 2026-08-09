# weatherbench.py
import os
import numpy as np
import pandas as pd

from tsl import logger
from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes import DatetimeDataset


class WeatherBench(DatetimeDataset):
    r"""WeatherBench: a benchmark dataset for data-driven weather
    forecasting, providing regridded ERA5 reanalysis data on a coarse
    global grid.

    Introduced in "WeatherBench: A benchmark dataset for data-driven
    weather forecasting" (Rasp et al., 2020, https://arxiv.org/abs/2002.00469).

    Each grid cell (lat/lon point) is treated as a node; the selected
    weather variable(s) are treated as channels.

    Args:
        root (str, optional): Root folder for data download/caching.
        variable (str): Which weather variable to load, e.g.
            'temperature', 'geopotential', '2m_temperature'.
            (default: '2m_temperature')
        resolution (str): Grid resolution, e.g. '5.625deg' (32x64 grid,
            the standard low-res WeatherBench setup) or '1.40625deg'.
            (default: '5.625deg')
        year_range (tuple, optional): (start_year, end_year) to subset
            the data. If None, loads the full available range.
            (default: None)
        freq (str, optional): Resampling frequency, e.g. '6H', '24H'.
            (default: '6H')
    """

    # Public WeatherBench data (Rasp et al.) is hosted on GCS as NetCDF/Zarr,
    # mirrored via the WeatherBench GitHub project. Update this base URL if it
    # moves - check https://github.com/pangeo-data/WeatherBench for current links.
    base_url = "https://dataserv.ub.tum.de/s/m1524895/download"

    similarity_options = {'distance'}

    def __init__(self,
                 root=None,
                 variable='2m_temperature',
                 resolution='5.625deg',
                 year_range=None,
                 freq='6H'):
        self.variable = variable
        self.resolution = resolution
        self.year_range = year_range
        self.root = root

        df, mask, lat_lon = self.load(freq=freq)

        super().__init__(target=df,
                          mask=mask,
                          freq=freq,
                          similarity_score='distance',
                          temporal_aggregation='mean',
                          spatial_aggregation='mean',
                          name='WeatherBench')

        # store grid coordinates as a static attribute, used later for
        # building a distance-based adjacency graph
        self.add_covariate('lat_lon', lat_lon, pattern='n c')

    @property
    def raw_file_names(self):
        return [f'{self.variable}_{self.resolution}.nc']

    @property
    def required_file_names(self):
        return self.raw_file_names

    def download(self) -> None:
        import urllib.request
        os.makedirs(self.root_dir, exist_ok=True)
        fname = self.raw_file_names[0]
        dest = os.path.join(self.root_dir, fname)
        if not os.path.exists(dest):
            logger.info(f"Downloading WeatherBench variable "
                        f"'{self.variable}' at {self.resolution} ...")
            # NOTE: verify the exact per-variable download path against
            # the current WeatherBench data repository before relying on
            # this in production - the public mirror's URL structure has
            # changed over the project's history.
            url = f"{self.base_url}/{fname}"
            urllib.request.urlretrieve(url, dest)
        else:
            logger.info("Raw file already present, skipping download.")

    def load_raw(self):
        import xarray as xr
        self.maybe_download()
        path = os.path.join(self.root_dir, self.raw_file_names[0])
        ds = xr.open_dataset(path)
        return ds

    def load(self, freq='6H'):
        ds = self.load_raw()

        # WeatherBench variables are typically stored as [time, lat, lon].
        # Flatten the spatial grid into a single "node" dimension.
        var_name = list(ds.data_vars)[0] if self.variable not in ds.data_vars \
            else self.variable
        da = ds[var_name]

        if self.year_range is not None:
            start, end = self.year_range
            da = da.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))

        lat = ds['lat'].values
        lon = ds['lon'].values
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        lat_lon = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

        values = da.values.reshape(da.shape[0], -1)  # [time, n_nodes]
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
            # haversine-ish approx: for a coarse regular grid, euclidean
            # distance on lat/lon is a reasonable, cheap proxy
            from scipy.spatial.distance import cdist
            dist = cdist(lat_lon, lat_lon, metric='euclidean')
            theta = np.std(dist)
            return gaussian_kernel(dist, theta=theta)
        raise NotImplementedError(f"Unknown similarity method '{method}'")