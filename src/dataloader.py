import os
import numpy as np
from tsl.datasets import MetrLA, PemsBay, ElectricityBenchmark
from tsl.datasets.pems_benchmarks import PeMS04, PeMS08
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.ops.connectivity import adj_to_edge_index


DATASET_MAP = {
    'metrla': MetrLA,
    'pemsbay': PemsBay,
    'pems04': PeMS04,
    'pems08': PeMS08,
    'electricity': ElectricityBenchmark,
}


DATASET_KWARGS = {}

CONN_DEFAULTS = {
    'electricity': {'threshold': 0.9, 'include_self': False, 'normalize_axis': 1, 'layout': 'edge_index'},
}


DEFAULT_WINDOW = {
    'electricity': 12,
}


def _correlation_connectivity(dataset, root, threshold=0.9, include_self=False,
                              normalize_axis=1, **kwargs):
    """Build a graph for datasets with no built-in similarity method
    (e.g. ElectricityBenchmark: similarity_options is None) from the
    absolute Pearson correlation between node series.

    The raw (pre-threshold, pre-normalize) correlation matrix is cached to
    disk under `root`, since np.corrcoef over the full dataframe is expensive
    (O(N^2 * T)) and would otherwise be recomputed on every call to
    get_dataloaders() -- i.e. every single trial. Thresholding/normalization
    are cheap and stay outside the cache so conn_threshold can still vary
    between calls without invalidating it.
    """
    cache_path = os.path.join(root, 'correlation_adj.npy')

    if os.path.exists(cache_path):
        adj = np.load(cache_path)
    else:
        values = dataset.dataframe().values
        adj = np.abs(np.corrcoef(values, rowvar=False))
        adj = np.nan_to_num(adj)
        np.save(cache_path, adj)

    adj = adj.copy()  # don't mutate the cached array in place

    if not include_self:
        np.fill_diagonal(adj, 0.0)

    adj[adj < threshold] = 0.0

    if normalize_axis is not None:
        denom = adj.sum(axis=normalize_axis, keepdims=True)
        denom[denom == 0] = 1.0
        adj = adj / denom

    return adj_to_edge_index(adj)


def get_dataloaders(
    dataset_name='metrla',
    window=None,
    horizon=12,
    batch_size=64,
    val_len=0.1,
    test_len=0.2,
    conn_threshold=0.1,
    base_root='./data',
    workers=None,
):
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_MAP:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_MAP.keys())}"
        )

    if window is None:
        window = DEFAULT_WINDOW.get(dataset_name, 12)

    root = os.path.join(base_root, dataset_name)
    os.makedirs(root, exist_ok=True)

    dataset_cls = DATASET_MAP[dataset_name]
    extra_kwargs = DATASET_KWARGS.get(dataset_name, {})
    dataset = dataset_cls(root=root, **extra_kwargs)

    conn_kwargs = CONN_DEFAULTS.get(dataset_name, {
        'threshold': conn_threshold,
        'include_self': False,
        'normalize_axis': 1,
        'layout': 'edge_index',
    })

    if dataset.similarity_options is None:
        # No built-in similarity method (e.g. ElectricityBenchmark and the
        # other tsl mts_benchmarks datasets) -> derive a graph from
        # correlation instead of dataset.get_connectivity().
        connectivity = _correlation_connectivity(dataset, root, **conn_kwargs)
    else:
        connectivity = dataset.get_connectivity(**conn_kwargs)

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        mask=dataset.mask,
        horizon=horizon,
        window=window,
        stride=1
    )

    scalers = {'target': StandardScaler(axis=(0, 1) if dataset_name != 'electricity' else (0,))}    
    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)

    if workers is None:
        workers = max(1, os.cpu_count() - 1)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=workers,
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    return train_loader, val_loader, test_loader