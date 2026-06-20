import os
import numpy as np # for array manipulation and .npz files 
import torch # convert data into tensors

# Resolve paths relative to this script's own location, not the terminal's
# working directory, so it works no matter where it's run from (VS Code's
# run button, a terminal in src/, a terminal at the project root, etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def create_windows(flow_data, input_length=12, horizon=3):
    """
    slices a (timesteps, num_sensors) array into (X, Y) training pairs

    input shape:
        flow_data           : shape (timesteps, num_sensors)
        input_length = 12   : how many past steps to use (input) 
        horizon = 3         : how far into the future to predict (3 = 15 minutes at 5-min intervals)

    output shape:
        X = shape (num_samples, input_length, num_sensors)
        Y = shape (num_samples, num_sensors)
    """
    
    num_timesteps, num_sensors = flow_data.shape # dimentions 

    # the last valid starting index, so that input_length + horizon
    # steps after it still fall inside the array
    last_start_index = num_timesteps - input_length - horizon

    # sample containers 
    X_list = []
    Y_list = []

    # sliding window loop 
    start_index = 0
    while start_index <= last_start_index:
        input_window = flow_data[start_index : start_index + input_length]
        target_index = start_index + input_length + horizon - 1
        target_value = flow_data[target_index]

        X_list.append(input_window)
        Y_list.append(target_value)

        start_index = start_index + 1

    # stack into arrays 
    X = np.stack(X_list)  # (num_samples, input_length, num_sensors)
    Y = np.stack(Y_list)  # (num_samples, num_sensors)

    return X, Y


def load_processed_dataset(processed_path, input_length=12, horizon=3):
    """
    loads a saved dataset file (.npz) and window it.
    """
    # load file
    data = np.load(processed_path)

    # create windows 
    train_X, train_Y = create_windows(data["train"], input_length, horizon)
    val_X, val_Y = create_windows(data["val"], input_length, horizon)
    test_X, test_Y = create_windows(data["test"], input_length, horizon)

    # convert to pytorch tensors and store in output dictionary
    tensors = {
        "train_X": torch.tensor(train_X, dtype=torch.float32),
        "train_Y": torch.tensor(train_Y, dtype=torch.float32),
        "val_X": torch.tensor(val_X, dtype=torch.float32),
        "val_Y": torch.tensor(val_Y, dtype=torch.float32),
        "test_X": torch.tensor(test_X, dtype=torch.float32),
        "test_Y": torch.tensor(test_Y, dtype=torch.float32),
        #normalisation stat
        "mean": float(data["mean"]), 
        "std": float(data["std"]),
    }

    return tensors


def load_adjacency(adjacency_path):
    """
    loads the adjacency matrix and returns a PyTorch tensor of shape (num_sensors, num_sensors).
    """
    data = np.load(adjacency_path) #load file
    adjacency = data["adjacency"] #extract matrix
    return torch.tensor(adjacency, dtype=torch.float32) # convert to tensor 


if __name__ == "__main__":
    # used to test 
    pems04_processed_path = os.path.join(DATA_DIR, "PEMS04_processed.npz")
    pems04_adjacency_path = os.path.join(DATA_DIR, "PEMS04_adjacency.npz")

    tensors = load_processed_dataset(pems04_processed_path)
    adjacency = load_adjacency(pems04_adjacency_path)

    print("train_X shape:", tensors["train_X"].shape)
    print("train_Y shape:", tensors["train_Y"].shape)
    print("val_X shape:", tensors["val_X"].shape)
    print("test_X shape:", tensors["test_X"].shape)
    print("adjacency shape:", adjacency.shape)