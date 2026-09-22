""" 
Loads the datasets and creates the train, validation and test dataloaders used across the experiments.
"""

import os
from tsl.datasets import MetrLA, PemsBay, AirQuality, ElectricityBenchmark
from tsl.datasets.pems_benchmarks import PeMS04, PeMS08
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter


# The available datasets are mapped to their corresponding TSL classes
DATASET_MAP = {
    'metrla': MetrLA,
    'pemsbay': PemsBay,
    'airquality': AirQuality,
    'electricity': ElectricityBenchmark,
    'pems04': PeMS04,
    'pems08': PeMS08,
}

def _correlation_connectivity(dataset, top_k=7, corr_threshold=0.9, cache_path=None):
    """Builds connectivity using the most correlated time series."""
    import numpy as np
    # The correlation matrix is cached to avoid recalculating it
    if cache_path is not None and os.path.exists(cache_path):
        adj = np.load(cache_path)
    else:
        adj = np.abs(np.corrcoef(dataset.dataframe().values, rowvar=False))
        adj = np.nan_to_num(adj)
        np.fill_diagonal(adj, 0.0)
        if cache_path is not None:
            np.save(cache_path, adj)

    if top_k is not None:
        # The most correlated series for each node are retained
        idx = np.argsort(-adj, axis=1)[:, :top_k]
        keep = np.zeros_like(adj)
        rows = np.arange(adj.shape[0])[:, None]
        keep[rows, idx] = adj[rows, idx]
        adj = keep
    else:
        adj[adj < corr_threshold] = 0.0

    # The edge weights are normalised across each row
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj = adj / row_sums

    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst], axis=0)
    edge_weight = adj[src, dst]
    return edge_index, edge_weight


def get_dataloaders(
    dataset_name='metrla',
    window=12,
    horizon=12,
    batch_size=64,
    val_len=0.1,
    test_len=0.2,
    conn_threshold=0.1,
    corr_top_k=6,
    base_root='./data',
    workers = None,
):
    """Returns the train, validation and test dataloaders for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_MAP.keys())}"
        )

    root = os.path.join(base_root, dataset_name)
    os.makedirs(root, exist_ok=True)

    dataset_cls = DATASET_MAP[dataset_name]
    dataset = dataset_cls(root=root)

    if dataset_name == 'electricity':
        connectivity = _correlation_connectivity(
            dataset,
            top_k=corr_top_k,
            cache_path=os.path.join(root, 'corr_matrix.npy'),
        )
    else:

        # The traffic graph is constructed from the road network distances
        connectivity = dataset.get_connectivity(
            threshold=conn_threshold,
            include_self=False,
            normalize_axis=1,
            layout='edge_index'
        )

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        # The mask identifies missing readings that are ignored by the loss
        mask=dataset.mask,
        horizon=horizon,
        window=window,
        stride=1
    )

    # Electricity series are standardised individually, while traffic sensors are standardised together
    scaler_axis = (0,) if dataset_name == 'electricity' else (0, 1)
    scalers = {'target': StandardScaler(axis=scaler_axis)}
    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)

    if workers == None:
        workers = max(1, os.cpu_count() - 1)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=workers,
    )

    # The scaler is fitted using the training data before the dataloaders are created
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    return train_loader, val_loader, test_loader