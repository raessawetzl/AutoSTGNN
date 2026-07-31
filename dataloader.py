import os
from tsl.datasets import MetrLA, PemsBay
from tsl.datasets.pems_benchmarks import PeMS04, PeMS08
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter


DATASET_MAP = {
    'metrla': MetrLA,
    'pemsbay': PemsBay,
    'pems04': PeMS04,
    'pems08': PeMS08,

}

def get_dataloaders(
    dataset_name='metrla',
    window=12,
    horizon=12,
    batch_size=64,
    val_len=0.1,
    test_len=0.2,
    conn_threshold=0.1,
    base_root='./data'
):
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

    connectivity = dataset.get_connectivity(
        threshold=conn_threshold,
        include_self=False,
        normalize_axis=1,
        layout='edge_index'
    )

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        mask=dataset.mask,
        horizon=horizon,
        window=window,
        stride=1
    )

    scalers = {'target': StandardScaler(axis=(0, 1))}
    splitter = TemporalSplitter(val_len=val_len, test_len=test_len)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=11,
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    return train_loader, val_loader, test_loader