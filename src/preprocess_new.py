import os
import numpy as np
from tsl.datasets import MetrLA, PemsBay

# Chronological split ratios 
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

DATASET_CONFIGS = [
    {
        "name": "METR-LA",
        "loader": MetrLA,
        "output_data_path": os.path.join(DATA_DIR, "METRLA_processed.npz"),
        "output_adj_path": os.path.join(DATA_DIR, "METRLA_adjacency.npz"),
    },
    {
        "name": "PEMS-Bay",
        "loader": PemsBay,
        "output_data_path": os.path.join(DATA_DIR, "PEMSBAY_processed.npz"),
        "output_adj_path": os.path.join(DATA_DIR, "PEMSBAY_adjacency.npz"),
    },
]


def split_chronologically(data, train_ratio, val_ratio):
    """
    Slices the time series into three chunks in chronological order: 
    train gets the earliest data (what the model learns from),
    val gets the middle period (used during hyperparameter search to score
    configurations), 
    test gets the most recent data (held back entirely and only touched at the very end to report final results)
    """
    num_timesteps = data.shape[0]
    train_end = int(round(num_timesteps * train_ratio))
    val_end = int(round(num_timesteps * (train_ratio + val_ratio)))

    train_data = data[0:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:num_timesteps]

    return train_data, val_data, test_data


def normalize_with_train_stats(train_data, val_data, test_data):
    """
    Computes a single global mean and standard deviation from the
    training data only, then applies that same scaling to train, val,
    and test. Using train-only stats avoids leaking information from
    the validation/test periods into the scaling

    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    normalized_train = (train_data - train_mean) / train_std
    normalized_val = (val_data - train_mean) / train_std
    normalized_test = (test_data - train_mean) / train_std

    return normalized_train, normalized_val, normalized_test, train_mean, train_std


def process_dataset(dataset_config, train_ratio, val_ratio):
    dataset_name = dataset_config["name"]
    loader_class = dataset_config["loader"]
    output_data_path = dataset_config["output_data_path"]
    output_adj_path = dataset_config["output_adj_path"]

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

    print("Loading " + dataset_name + " from tsl...")
    dataset = loader_class(root=DATA_DIR)

    # Pull out the speed readings as a plain numpy array. Rows are timesteps, columns are sensors
    data = dataset.dataframe().values.astype(np.float32)

    # Get adjacency matrix from tsl's built-in similarity computation
    # Uses thresholded Gaussian kernel on pairwise road network distances
    adj_matrix = dataset.get_similarity().astype(np.float32)

    print("Data shape: " + str(data.shape))
    print("Adjacency matrix shape: " + str(adj_matrix.shape))
    print("Non-zero edges: " + str(int((adj_matrix > 0).sum())))

    # Split chronologically
    train_data, val_data, test_data = split_chronologically(data, train_ratio, val_ratio)

    # Normalize using training stats only
    normalized_train, normalized_val, normalized_test, train_mean, train_std = normalize_with_train_stats(
        train_data, val_data, test_data
    )

    # Save processed data
    output_file = open(output_data_path, "wb")
    np.savez(
        output_file,
        train=normalized_train,
        val=normalized_val,
        test=normalized_test,
        mean=train_mean,
        std=train_std,
    )
    output_file.close()

    # Save adjacency matrix
    adj_file = open(output_adj_path, "wb")
    np.savez(adj_file, adjacency=adj_matrix)
    adj_file.close()

    print("Train shape: " + str(normalized_train.shape))
    print("Val shape: " + str(normalized_val.shape))
    print("Test shape: " + str(normalized_test.shape))
    print("Train mean: " + str(round(float(train_mean), 4)) + " | Train std: " + str(round(float(train_std), 4)))
    print("Saved processed data to " + output_data_path)
    print("Saved adjacency matrix to " + output_adj_path)
    print("")


def main():
    config_index = 0
    while config_index < len(DATASET_CONFIGS):
        process_dataset(DATASET_CONFIGS[config_index], TRAIN_RATIO, VAL_RATIO)
        config_index = config_index + 1


if __name__ == "__main__":
    main()