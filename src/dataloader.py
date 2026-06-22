import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Configuration
#
# window: number of past timesteps fed to the model as input (12 steps
# at 5-minute resolution = 1 hour of history)
# horizon: number of future timesteps the model must predict (12 steps
# = 1 hour ahead)
WINDOW = 12
HORIZON = 12
BATCH_SIZE = 64

DATASET_PATHS = {
    "PEMS04": {
        "processed_path": "PEMS04_processed.npz",
        "adjacency_path": "PEMS04_adjacency.npz",
    },
    "PEMS08": {
        "processed_path": "PEMS08_processed.npz",
        "adjacency_path": "PEMS08_adjacency.npz",
    },
}


def add_window_horizon(flow_split, window, horizon):
    """
    flow_split: array of shape (timesteps, sensors)
    Returns:
        X: array of shape (num_samples, window, sensors, 1)
        Y: array of shape (num_samples, horizon, sensors, 1)

    """
    num_timesteps = flow_split.shape[0]
    last_valid_start = num_timesteps - window - horizon + 1

    X = []
    Y = []
    start_index = 0
    while start_index < last_valid_start:
        x_sample = flow_split[start_index: start_index + window]
        y_sample = flow_split[start_index + window: start_index + window + horizon]
        X.append(x_sample)
        Y.append(y_sample)
        start_index = start_index + 1

    X = np.array(X)
    Y = np.array(Y)

    # add the channel dimension: (samples, T, N) -> (samples, T, N, 1)
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    return X, Y


def build_dataloader(X, Y, batch_size, shuffle, drop_last):
    """
    Wraps windowed numpy arrays into a PyTorch DataLoader.
    """
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def load_adjacency_matrix(adjacency_path):
    """
    Loads precomputed adjacency matrix
    """
    adjacency_file = open(adjacency_path, "rb")
    adjacency_data = np.load(adjacency_file)
    adj_mx = adjacency_data["adjacency"]
    adjacency_file.close()
    return adj_mx


def get_dataloaders(dataset_name, window=WINDOW, horizon=HORIZON, batch_size=BATCH_SIZE):
    """
    Main entry point. Loads the processed train/val/test flow splits,
    windows each split independently (so no sample crosses a split
    boundary), and returns ready-to-use DataLoaders plus the
    normalization stats and adjacency matrix.

    Returns:
        train_loader, val_loader, test_loader, mean, std, adj_mx
    """
    paths = DATASET_PATHS[dataset_name]

    processed_file = open(paths["processed_path"], "rb")
    processed_data = np.load(processed_file)
    train_data = processed_data["train"]
    val_data = processed_data["val"]
    test_data = processed_data["test"]
    mean = float(processed_data["mean"])
    std = float(processed_data["std"])
    processed_file.close()

    x_train, y_train = add_window_horizon(train_data, window, horizon)
    x_val, y_val = add_window_horizon(val_data, window, horizon)
    x_test, y_test = add_window_horizon(test_data, window, horizon)

    train_loader = build_dataloader(x_train, y_train, batch_size, shuffle=True, drop_last=True)
    val_loader = build_dataloader(x_val, y_val, batch_size, shuffle=False, drop_last=True)
    test_loader = build_dataloader(x_test, y_test, batch_size, shuffle=False, drop_last=False)

    adj_mx = load_adjacency_matrix(paths["adjacency_path"])

    print("--- " + dataset_name + " ---")
    print("Train samples: " + str(x_train.shape[0]) + ", shape per sample: " + str(x_train.shape[1:]))
    print("Val samples: " + str(x_val.shape[0]))
    print("Test samples: " + str(x_test.shape[0]))
    print("Adjacency matrix shape: " + str(adj_mx.shape))
    print("")

    return train_loader, val_loader, test_loader, mean, std, adj_mx


if __name__ == "__main__":
    train_loader, val_loader, test_loader, mean, std, adj_mx = get_dataloaders("PEMS04")

    # quick check: pull one batch and print its shape
    x_batch, y_batch = next(iter(train_loader))
    print("Example batch X shape:", x_batch.shape)  # (B, window, N, 1)
    print("Example batch Y shape:", y_batch.shape)  # (B, horizon, N, 1)