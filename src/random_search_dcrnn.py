import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from models.dcrnn_model import DCRNNModel
from search_space import get_shared_config_space

# Runs random hyperparameter search on DCRNN as a naive baseline.
# Each trial samples a random configuration from the search space and
# trains DCRNN with those values. The best configuration found across
# all trials is recorded. This serves as the comparison point for DEHB —
# can DEHB find a better configuration than random search within the
# same time budget?

# Number of random configurations to try. Each one is a full training run.
# On the HPC cluster this is feasible; on a laptop reduce to 2-3 for testing.
NUM_TRIALS = 20

# Number of epochs per trial. Keep low for testing, use 100 for real results.
EPOCHS_PER_TRIAL = 100

SEQ_LEN = 12        # number of historical timesteps fed into the model (1 hour)
HORIZON = 3         # number of future timesteps to predict (15 minutes)
INPUT_DIM = 1       # only flow is used
OUTPUT_DIM = 1      # only flow is predicted
MAX_DIFFUSION_STEP = 2
FILTER_TYPE = 'random_walk'
CL_DECAY_STEPS = 1000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

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


def get_dcrnn_config_space():
    """
    DCRNN search space = shared hyperparameters + max_diffusion_step.
    max_diffusion_step controls how many hops the diffusion process
    spreads across the graph — more steps = wider spatial context.
    """
    cs = get_shared_config_space()
    cs.add(
        CSH.UniformIntegerHyperparameter(
            name='max_diffusion_step',
            lower=1,
            upper=4
        )
    )
    return cs


def setup_logger():
    logging.basicConfig(level=logging.WARNING)
    return logging.getLogger("DCRNN")


def load_data(data_path):
    """
    Loads the preprocessed train, val and test arrays produced by
    preprocess.py, along with the mean and std used during normalisation.
    The mean and std are needed later to convert predictions back to
    real vehicle counts when computing MAE, RMSE and MAPE.
    """
    data_file = np.load(data_path)
    return (data_file["train"], data_file["val"], data_file["test"],
            float(data_file["mean"]), float(data_file["std"]))


def load_adjacency(adj_path):
    """
    Loads the adjacency matrix produced by build_adjacency.py and
    converts it to a PyTorch tensor so the DCRNN model can use it
    during training to model spatial relationships between sensors.
    """
    adj = np.load(adj_path)["adjacency"].astype(np.float32)
    return torch.tensor(adj, device=device)


def create_windows(data, seq_len, horizon):
    """
    Creates (input, target) pairs from the time series. The model uses
    the input timesteps to predict the target timesteps, which are actual
    recorded sensor readings.
    """
    inputs, targets = [], []
    i = 0
    while i + seq_len + horizon <= data.shape[0]:
        inputs.append(data[i: i + seq_len])
        targets.append(data[i + seq_len: i + seq_len + horizon])
        i = i + 1
    return np.array(inputs), np.array(targets)


def make_batches(inputs, targets, batch_size):
    """
    Takes the (input, target) pairs from create_windows() and groups them
    into smaller batches so the model does not process all examples at once,
    which would use too much memory. Examples are shuffled randomly before
    batching so the model sees them in a different order each epoch, helping
    it learn more generalised patterns.
    """
    indices = np.random.permutation(inputs.shape[0])
    i = 0
    while i < inputs.shape[0]:
        idx = indices[i: i + batch_size]
        yield (
            torch.tensor(inputs[idx], dtype=torch.float32, device=device),
            torch.tensor(targets[idx], dtype=torch.float32, device=device)
        )
        i = i + batch_size


def prepare_model_input(batch):
    """
    Reshapes a batch from (batch_size, seq_len, num_nodes) to the shape
    DCRNNModel expects: (seq_len, batch_size, num_nodes * input_dim).
    """
    batch_size, seq_len, num_nodes = batch.shape
    return batch.permute(1, 0, 2).reshape(seq_len, batch_size, num_nodes * INPUT_DIM)


def compute_metrics(predictions, targets, mean, std):
    """
    Converts normalised predictions and targets back to real vehicle counts
    using the mean and std saved during preprocessing, then computes three
    error metrics: MAE, RMSE, and MAPE. Sensors with fewer than 10 vehicles
    are excluded from MAPE to avoid division by near-zero values.
    """
    predictions_real = predictions * std + mean
    targets_real = targets * std + mean
    mae = np.mean(np.abs(predictions_real - targets_real))
    rmse = np.sqrt(np.mean((predictions_real - targets_real) ** 2))
    mask = targets_real > 10
    mape = np.mean(np.abs((predictions_real[mask] - targets_real[mask]) / targets_real[mask])) * 100
    return mae, rmse, mape


def run_trial(config, dataset_config, logger, trial_num):
    """
    Trains DCRNN with one randomly sampled hyperparameter configuration
    for EPOCHS_PER_TRIAL epochs. Evaluates on validation after each epoch
    and saves the best model weights. Returns the best validation MAE and
    the final test MAE, RMSE and MAPE from the best saved model.
    """
    num_nodes = dataset_config["num_nodes"]
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    hidden_units = int(config["hidden_units"])
    num_layers = int(config["num_layers"])
    max_diffusion_step = int(config["max_diffusion_step"])
    weight_decay = float(config["weight_decay"])

    train_data, val_data, test_data, mean, std = load_data(dataset_config["data_path"])
    adj_matrix = load_adjacency(dataset_config["adj_path"])

    train_inputs, train_targets = create_windows(train_data, SEQ_LEN, HORIZON)
    val_inputs, val_targets = create_windows(val_data, SEQ_LEN, HORIZON)
    test_inputs, test_targets = create_windows(test_data, SEQ_LEN, HORIZON)

    model_kwargs = {
        "num_nodes": num_nodes,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "rnn_units": hidden_units,
        "num_rnn_layers": num_layers,
        "max_diffusion_step": max_diffusion_step,
        "filter_type": FILTER_TYPE,
        "cl_decay_steps": CL_DECAY_STEPS,
        "use_curriculum_learning": True,
    }

    model = DCRNNModel(adj_matrix, logger, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    batches_seen = 0

    epoch = 0
    while epoch < EPOCHS_PER_TRIAL:
        model.train()
        for input_batch, target_batch in make_batches(train_inputs, train_targets, batch_size):
            model_input = prepare_model_input(input_batch)
            b, _, n = input_batch.shape
            model_target = target_batch.permute(1, 0, 2).reshape(HORIZON, b, n * OUTPUT_DIM)
            optimizer.zero_grad()
            output = model(model_input, model_target, batches_seen)
            loss = criterion(output, model_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batches_seen = batches_seen + 1

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for input_batch, target_batch in make_batches(val_inputs, val_targets, batch_size):
                model_input = prepare_model_input(input_batch)
                b, _, n = input_batch.shape
                output = model(model_input).reshape(HORIZON, b, n).permute(1, 0, 2)
                all_preds.append(output.cpu().numpy())
                all_targets.append(target_batch.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        tgts = np.concatenate(all_targets, axis=0)
        val_mae, _, _ = compute_metrics(preds, tgts, mean, std)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "dcrnn_trial_best.pt")

        epoch = epoch + 1

    model.load_state_dict(torch.load("dcrnn_trial_best.pt"))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for input_batch, target_batch in make_batches(test_inputs, test_targets, batch_size):
            model_input = prepare_model_input(input_batch)
            b, _, n = input_batch.shape
            output = model(model_input).reshape(HORIZON, b, n).permute(1, 0, 2)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_targets, axis=0)
    test_mae, test_rmse, test_mape = compute_metrics(preds, tgts, mean, std)

    return best_val_mae, test_mae, test_rmse, test_mape


def run_random_search(dataset_config):
    """
    Pipeline manager for random search on one dataset. Samples NUM_TRIALS
    random configurations from the search space, trains DCRNN with each,
    and records the best configuration found based on validation MAE.
    Saves a JSON log of all trials to the results directory.
    """
    dataset_name = dataset_config["name"]
    logger = setup_logger()
    cs = get_dcrnn_config_space()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "dcrnn_" + dataset_name + "_random_search.json")

    print("=" * 50)
    print("Random search: DCRNN on " + dataset_name)
    print("Trials: " + str(NUM_TRIALS) + " | Epochs per trial: " + str(EPOCHS_PER_TRIAL))
    print("=" * 50)

    search_log = []
    best_val_mae = float("inf")
    best_config = None
    best_test_metrics = None

    trial = 0
    while trial < NUM_TRIALS:
        config = cs.sample_configuration()
        config_dict = dict(config)

        print("Trial " + str(trial + 1) + "/" + str(NUM_TRIALS) + " | Config: " + str(config_dict))

        val_mae, test_mae, test_rmse, test_mape = run_trial(config_dict, dataset_config, logger, trial)

        print("  Val MAE: " + str(round(val_mae, 4)) +
              " | Test MAE: " + str(round(test_mae, 4)) +
              " | Test RMSE: " + str(round(test_rmse, 4)) +
              " | Test MAPE: " + str(round(test_mape, 4)) + "%")

        log_entry = {
            "trial": trial + 1,
            "config": config_dict,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
        }
        search_log.append(log_entry)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = config_dict
            best_test_metrics = {"mae": test_mae, "rmse": test_rmse, "mape": test_mape}
            print("  *** New best configuration found ***")

        with open(log_path, "w") as f:
            json.dump(search_log, f, indent=2)

        trial = trial + 1

    print("")
    print("Best configuration for " + dataset_name + ":")
    print(json.dumps(best_config, indent=2))
    print("Best val MAE: " + str(round(best_val_mae, 4)))
    print("Test MAE: " + str(round(best_test_metrics["mae"], 4)))
    print("Test RMSE: " + str(round(best_test_metrics["rmse"], 4)))
    print("Test MAPE: " + str(round(best_test_metrics["mape"], 4)) + "%")
    print("Search log saved to " + log_path)

    return best_config, best_val_mae, best_test_metrics


def main():
    # Starts the random search pipeline for each dataset in DATASET_CONFIGS
    config_index = 0
    while config_index < len(DATASET_CONFIGS):
        run_random_search(DATASET_CONFIGS[config_index])
        config_index = config_index + 1


if __name__ == "__main__":
    main()