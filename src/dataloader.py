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

# k: number of nearest neighbours (by abs. Pearson correlation) each node
# keeps an edge to. Replaces the fixed threshold, which left 164/321 (51%)
# of Electricity's nodes fully isolated -- no threshold value transfers
# well across such a heterogeneous set of correlations, whereas k-NN
# guarantees every node gets exactly k neighbours regardless of its
# absolute correlation level.
CONN_DEFAULTS = {
    'electricity': {'k': 6, 'include_self': False, 'layout': 'edge_index'},
}


DEFAULT_WINDOW = {
    'electricity': 12,
}


def _knn_connectivity(dataset, root, k=6, include_self=False, **kwargs):
    """Build a k-NN graph for datasets with no built-in similarity method
    (e.g. ElectricityBenchmark: similarity_options is None), from the
    absolute Pearson correlation between node series.

    Each node keeps an edge to its k highest-correlation neighbours. The
    result is symmetrized (union: an edge is kept if either endpoint picked
    the other), since a directed top-k graph is not what tsl's GNN layers
    expect, and mutual-kNN (both endpoints must pick each other) would leave
    some nodes isolated again -- the exact problem being fixed here.

    The raw correlation matrix is cached to disk under `root`, since
    np.corrcoef over the full dataframe is expensive (O(N^2 * T)) and would
    otherwise be recomputed on every call to get_dataloaders() -- i.e. every
    single trial. k can still vary between calls without invalidating the
    cache, since only the (cheap) top-k selection depends on it.
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
    n = adj.shape[0]

    if not include_self:
        np.fill_diagonal(adj, -np.inf)  # never selected as a neighbour

    if k >= n - 1:
        raise ValueError(f"k={k} must be smaller than n_nodes-1={n - 1}")

    # top-k neighbour indices per row (unsorted within the top-k, order
    # doesn't matter since we only need the *set* of kept edges)
    topk_idx = np.argpartition(-adj, kth=k, axis=1)[:, :k]

    knn_mask = np.zeros_like(adj, dtype=bool)
    rows = np.repeat(np.arange(n), k)
    cols = topk_idx.ravel()
    knn_mask[rows, cols] = True

    # union: keep an edge if either endpoint selected the other, so a node
    # is only isolated if it appears in *no* row's top-k anywhere -- far
    # less likely than requiring a fixed absolute correlation threshold
    knn_mask = knn_mask | knn_mask.T

    if not include_self:
        np.fill_diagonal(knn_mask, False)

    out_adj = np.where(knn_mask, np.abs(adj), 0.0)
    # restore true self-correlation (1.0) if include_self was requested,
    # since it was set to -inf above purely to exclude it from top-k search
    if include_self:
        np.fill_diagonal(out_adj, 1.0)

    return adj_to_edge_index(out_adj)


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
        # correlation-based k-NN instead of dataset.get_connectivity().
        connectivity = _knn_connectivity(dataset, root, **conn_kwargs)
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