import os
import logging
import numpy as np
import torch
import torch.nn as nn

from models.dcrnn_model import DCRNNModel


# Trains DCRNN on both datasets using fixed default hyperparameters
# This is the baseline run — it represents what DCRNN achieves without
# any hyperparameter tuning. DEHB and random search are expected to
# improve on these results

SEQ_LEN = 12   # he number of historical timesteps fed into the model as input
HORIZON = 3  # number of future timesteps to predict (15 minutes)
INPUT_DIM = 1  # only flow is used
OUTPUT_DIM = 1   # only flow is predicted
RNN_UNITS = 64      # size of hidden layers
NUM_RNN_LAYERS = 2  # number of GRU layers
MAX_DIFFUSION_STEP = 2
FILTER_TYPE = 'random_walk'
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_EPOCHS = 2
CL_DECAY_STEPS = 1000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

DATASET_CONFIGS = [
    {
        "name": "PEMS04",
        "data_path": os.path.join(DATA_DIR, "PEMS04_processed.npz"),
        "adj_path": os.path.join(DATA_DIR, "PEMS04_adjacency.npz"),
        "num_nodes": 307,
    },
    {
        "name": "PEMS08",
        "data_path": os.path.join(DATA_DIR, "PEMS08_processed.npz"),
        "adj_path": os.path.join(DATA_DIR, "PEMS08_adjacency.npz"),
        "num_nodes": 170,
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger():
    """Sets up a simple logger so DCRNNModel's internal logging works"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("DCRNN")


def load_data(data_path):
    """
    Loads the preprocessed train, val and test arrays produced by
    preprocess.py, along with the mean and std used during normalisation

    The mean and std are needed later to convert predictions back to
    real vehicle counts when computing MAE, RMSE and MAPE

    """
    data_file = np.load(data_path)
    train = data_file["train"]
    val = data_file["val"]
    test = data_file["test"]
    mean = float(data_file["mean"])
    std = float(data_file["std"])
    return train, val, test, mean, std


def load_adjacency(adj_path):
    """
    Loads the adjacency matrix produced by build_adjacency.py and
    converts it to a PyTorch tensor so the DCRNN model can use it
    during training to model spatial relationships between sensors

    """
    adj_file = np.load(adj_path)
    adj_matrix = adj_file["adjacency"].astype(np.float32)
    return torch.tensor(adj_matrix, device=device)


def create_windows(data, seq_len, horizon):
    """
    Creates (input, target) pairs from the time series. The model uses
    the input timesteps to predict the target timesteps, which are actual
    recorded sensor readings
    
    """
    num_timesteps = data.shape[0]
    inputs = []
    targets = []

    window_index = 0
    while window_index + seq_len + horizon <= num_timesteps:
        input_window = data[window_index: window_index + seq_len]
        target_window = data[window_index + seq_len: window_index + seq_len + horizon]
        inputs.append(input_window)
        targets.append(target_window)
        window_index = window_index + 1

    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets


def make_batches(inputs, targets, batch_size):
    """
    Takes the (input, target) pairs from create_windows() and groups them
    into smaller batches so the model does not process all examples at once,
    which would use too much memory. Examples are shuffled randomly before
    batching so the model sees them in a different order each epoch, helping
    it learn more generalised patterns
     
    """
    num_samples = inputs.shape[0]
    indices = np.random.permutation(num_samples)
    batch_start = 0
    while batch_start < num_samples:
        batch_indices = indices[batch_start: batch_start + batch_size]
        input_batch = torch.tensor(inputs[batch_indices], dtype=torch.float32, device=device)
        target_batch = torch.tensor(targets[batch_indices], dtype=torch.float32, device=device)
        yield input_batch, target_batch
        batch_start = batch_start + batch_size


def prepare_model_input(batch):
    """
    Reshapes a batch from (batch_size, seq_len, num_nodes) to the shape
    DCRNNModel expects: (seq_len, batch_size, num_nodes * input_dim)

    """
    batch_size, seq_len, num_nodes = batch.shape
    return batch.permute(1, 0, 2).reshape(seq_len, batch_size, num_nodes * INPUT_DIM)


def compute_metrics(predictions, targets, mean, std):
    """
    Converts normalised predictions and targets back to real vehicle counts
    using the mean and std saved during preprocessing, then computes three
    error metrics: MAE, RMSE, and MAPE
    
    """
    predictions_real = predictions * std + mean
    targets_real = targets * std + mean

    mae = np.mean(np.abs(predictions_real - targets_real))
    rmse = np.sqrt(np.mean((predictions_real - targets_real) ** 2))

    nonzero_mask = targets_real > 10
    mape = np.mean(np.abs((predictions_real[nonzero_mask] - targets_real[nonzero_mask]) / targets_real[nonzero_mask])) * 100

    return mae, rmse, mape


def train_one_epoch(model, optimizer, criterion, inputs, targets, batches_seen):
    """
    Runs one full pass over the training batches. For each batch the model
     makes predictions, calculates the loss (difference between predictions
    and actual values), and updates its weights to reduce that loss
    Returns the average loss across all batches for that epoch
    
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for input_batch, target_batch in make_batches(inputs, targets, BATCH_SIZE):
        model_input = prepare_model_input(input_batch)
        batch_size = input_batch.shape[0]
        num_nodes = input_batch.shape[2]

        model_target = target_batch.permute(1, 0, 2).reshape(HORIZON, batch_size, num_nodes * OUTPUT_DIM)

        optimizer.zero_grad()
        output = model(model_input, model_target, batches_seen)
        loss = criterion(output, model_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss = total_loss + loss.item()
        num_batches = num_batches + 1
        batches_seen = batches_seen + 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, batches_seen


def evaluate(model, inputs, targets, mean, std):
    """
    Runs the model on val or test data with fixed weights from training.
    No weight updates happen here — the model just makes predictions on
    unseen data and returns the MAE, RMSE and MAPE error scores

    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for input_batch, target_batch in make_batches(inputs, targets, BATCH_SIZE):
            model_input = prepare_model_input(input_batch)
            batch_size = input_batch.shape[0]
            num_nodes = input_batch.shape[2]

            output = model(model_input)
            output = output.reshape(HORIZON, batch_size, num_nodes).permute(1, 0, 2)

            all_predictions.append(output.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae, rmse, mape = compute_metrics(all_predictions, all_targets, mean, std)
    return mae, rmse, mape


def train_dataset(dataset_config, logger):
    """
    Pipeline manager for one dataset. It calls all functions in sequence:
    loads data and adjacency matrix, creates windows, builds the model,
    trains for MAX_EPOCHS, and evaluates the best model on test data
    Returns the final MAE, RMSE and MAPE

    """
    dataset_name = dataset_config["name"]
    num_nodes = dataset_config["num_nodes"]

    print("=" * 50)
    print("Training DCRNN on " + dataset_name)
    print("=" * 50)

    train_data, val_data, test_data, mean, std = load_data(dataset_config["data_path"])
    adj_matrix = load_adjacency(dataset_config["adj_path"])

    train_inputs, train_targets = create_windows(train_data, SEQ_LEN, HORIZON)
    val_inputs, val_targets = create_windows(val_data, SEQ_LEN, HORIZON)
    test_inputs, test_targets = create_windows(test_data, SEQ_LEN, HORIZON)

    print("Train windows: " + str(train_inputs.shape[0]))
    print("Val windows: " + str(val_inputs.shape[0]))
    print("Test windows: " + str(test_inputs.shape[0]))

    model_kwargs = {
        "num_nodes": num_nodes,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "rnn_units": RNN_UNITS,
        "num_rnn_layers": NUM_RNN_LAYERS,
        "max_diffusion_step": MAX_DIFFUSION_STEP,
        "filter_type": FILTER_TYPE,
        "cl_decay_steps": CL_DECAY_STEPS,
        "use_curriculum_learning": True,
    }

    model = DCRNNModel(adj_matrix, logger, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    batches_seen = 0

    epoch = 0
    while epoch < MAX_EPOCHS:
        train_loss, batches_seen = train_one_epoch(
            model, optimizer, criterion, train_inputs, train_targets, batches_seen
        )
        val_mae, val_rmse, val_mape = evaluate(model, val_inputs, val_targets, mean, std)

        print("Epoch " + str(epoch + 1) + "/" + str(MAX_EPOCHS) +
              " | Train loss: " + str(round(train_loss, 4)) +
              " | Val MAE: " + str(round(val_mae, 4)) +
              " | Val RMSE: " + str(round(val_rmse, 4)) +
              " | Val MAPE: " + str(round(val_mape, 4)) + "%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), dataset_name + "_dcrnn_best.pt")

        epoch = epoch + 1

    model.load_state_dict(torch.load(dataset_name + "_dcrnn_best.pt"))
    test_mae, test_rmse, test_mape = evaluate(model, test_inputs, test_targets, mean, std)

    print("")
    print("Final Test Results for " + dataset_name + ":")
    print("MAE:  " + str(round(test_mae, 4)))
    print("RMSE: " + str(round(test_rmse, 4)))
    print("MAPE: " + str(round(test_mape, 4)) + "%")
    print("")

    return test_mae, test_rmse, test_mape


def main():
    logger = setup_logger()
    results = []

    config_index = 0
    while config_index < len(DATASET_CONFIGS):
        dataset_config = DATASET_CONFIGS[config_index]
        test_mae, test_rmse, test_mape = train_dataset(dataset_config, logger)
        results.append({
            "dataset": dataset_config["name"],
            "mae": test_mae,
            "rmse": test_rmse,
            "mape": test_mape,
        })
        config_index = config_index + 1

    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print("{:<12} {:<10} {:<10} {:<10}".format("Dataset", "MAE", "RMSE", "MAPE"))
    result_index = 0
    while result_index < len(results):
        r = results[result_index]
        print("{:<12} {:<10} {:<10} {:<10}".format(
            r["dataset"],
            round(r["mae"], 4),
            round(r["rmse"], 4),
            str(round(r["mape"], 4)) + "%"
        ))
        result_index = result_index + 1


if __name__ == "__main__":
    main()

