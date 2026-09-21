import os
import torch
import numpy as np
from tsl.datasets import MetrLA, PemsBay, LargeST
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter

WINDOW = 12
HORIZON = 12

DATASET_MAP = {
    'metr-la':  MetrLA,
    'pems-bay': PemsBay,
    'largeST':  LargeST,
}

def get_dataloaders(
    dataset_name,
    window=WINDOW,
    horizon=HORIZON,
    batch_size=64,
    root='./data'
):
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in DATASET_MAP:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_MAP.keys())}")

    root = os.path.join(root, dataset_name_lower)
    os.makedirs(root, exist_ok=True)

    dataset_cls = DATASET_MAP[dataset_name_lower]
    dataset = dataset_cls(root=root)

    connectivity = dataset.get_connectivity(
        threshold=0.1,
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
    splitter = TemporalSplitter(val_len=0.2, test_len=0.1)  # 7:2:1 split

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=4,
    )
    dm.setup()

    print(f"--- {dataset_name} ---")
    print(f"Train samples: {len(dm.train_dataloader().dataset)}")
    print(f"Val samples:   {len(dm.val_dataloader().dataset)}")
    print(f"Test samples:  {len(dm.test_dataloader().dataset)}")
    print()

    return (
        dm.train_dataloader(),
        dm.val_dataloader(shuffle=False),
        dm.test_dataloader(),
        dm.scalers['target'],
    )