import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from dehb import DEHB
import ConfigSpace.hyperparameters as CSH

from models.dcrnn_model import DCRNNModel
from search_space import get_shared_config_space

# Configuration

# This script uses DEHB (Differential Evolution Hyperband) to automatically
# search for the best hyperparameter configuration for DCRNN within a fixed
# time budget. Unlike random search which picks configurations blindly, DEHB
# uses multi-fidelity optimisation to focus compute on promising configurations
# and discard poor ones early, making it more efficient than random search

MIN_FIDELITY = 2   # minimum epochs per trial. DEHB starts here for cheap evaluations
MAX_FIDELITY = 3  # maximum epochs per trial (full evaluation) , only promising configurations reach this


RUNTIME_SECONDS = 3600 # total time/ wallclock budget for the DEHB search per dataset

SEQ_LEN = 12
HORIZON = 3
INPUT_DIM = 1
OUTPUT_DIM = 1
FILTER_TYPE = 'random_walk'
CL_DECAY_STEPS = 1000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

DATASET_CONFIGS = [
    {
        "name": "METR-LA",
        "data_path": os.path.join(DATA_DIR, "METRLA_processed.npz"),
        "adj_path": os.path.join(DATA_DIR, "METRLA_adjacency.npz"),
        "num_nodes": 207,
    },
    {
        "name": "PEMS-Bay",
        "data_path": os.path.join(DATA_DIR, "PEMSBAY_processed.npz"),
        "adj_path": os.path.join(DATA_DIR, "PEMSBAY_adjacency.npz"),
        "num_nodes": 325,
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dcrnn_config_space():
    """
    ConfigSpace defines the range of values each hyperparameter can be
    tuned to. The shared space covers hyperparameters common to all three
    models. This function extends that shared space by adding
    max_diffusion_step, which is specific to DCRNN

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
    data_file = np.load(data_path)
    return (data_file["train"], data_file["val"], data_file["test"],
            float(data_file["mean"]), float(data_file["std"]))


def load_adjacency(adj_path):
    adj = np.load(adj_path)["adjacency"].astype(np.float32)
    return torch.tensor(adj, device=device)


def create_windows(data, seq_len, horizon):
    inputs, targets = [], []
    i = 0
    while i + seq_len + horizon <= data.shape[0]:
        inputs.append(data[i: i + seq_len])
        targets.append(data[i + seq_len: i + seq_len + horizon])
        i = i + 1
    return np.array(inputs), np.array(targets)


def make_batches(inputs, targets, batch_size):
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
    batch_size, seq_len, num_nodes = batch.shape
    return batch.permute(1, 0, 2).reshape(seq_len, batch_size, num_nodes * INPUT_DIM)


def compute_mae(predictions, targets, mean, std):
    """
    Converts normalized predictions and targets back to real vehicle
    counts and computes MAE. Used as the fitness value for DEHB —
    lower is better.
    """
    predictions_real = predictions * std + mean
    targets_real = targets * std + mean
    return float(np.mean(np.abs(predictions_real - targets_real)))


def compute_metrics(predictions, targets, mean, std):
    predictions_real = predictions * std + mean
    targets_real = targets * std + mean
    mae = np.mean(np.abs(predictions_real - targets_real))
    rmse = np.sqrt(np.mean((predictions_real - targets_real) ** 2))
    mask = targets_real > 10
    mape = np.mean(np.abs((predictions_real[mask] - targets_real[mask]) / targets_real[mask])) * 100
    return float(mae), float(rmse), float(mape)


def train_and_evaluate(config, fidelity, dataset_config, logger):
    """
    Trains DCRNN with the given hyperparameter configuration for the
    specified number of epochs (fidelity) and returns the validation MAE
    and training cost. DEHB uses these to decide whether the configuration
    is promising enough to promote to a higher fidelity

    """
    num_nodes = dataset_config["num_nodes"]
    train_data, val_data, test_data, mean, std = load_data(dataset_config["data_path"])
    adj_matrix = load_adjacency(dataset_config["adj_path"])

    train_inputs, train_targets = create_windows(train_data, SEQ_LEN, HORIZON)
    val_inputs, val_targets = create_windows(val_data, SEQ_LEN, HORIZON)

    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    hidden_units = int(config["hidden_units"])
    num_layers = int(config["num_layers"])
    max_diffusion_step = int(config["max_diffusion_step"])
    weight_decay = float(config["weight_decay"])
    num_epochs = int(fidelity)

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

    start_time = time.time()
    batches_seen = 0

    epoch = 0
    while epoch < num_epochs:
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
        epoch = epoch + 1

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
    val_mae = compute_mae(preds, tgts, mean, std)
    cost = time.time() - start_time

    return val_mae, cost, model, test_data, mean, std


def run_dehb(dataset_config):

    """
    Pipeline manager for the DEHB search on one dataset. Sets up the
    search space, initialises DEHB with the target function, and runs
    the search within the time budget. Tracks the best configuration
    found, saves a JSON log of all trials, and evaluates the best model
    on test data at the end

    """
    dataset_name = dataset_config["name"]
    logger = setup_logger()
    cs = get_dcrnn_config_space()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "dcrnn_" + dataset_name + "_dehb.json")
    dehb_log_dir = os.path.join(RESULTS_DIR, "dehb_logs_" + dataset_name)

    print("=" * 50)
    print("DEHB optimization: DCRNN on " + dataset_name)
    print("Min fidelity: " + str(MIN_FIDELITY) + " epochs")
    print("Max fidelity: " + str(MAX_FIDELITY) + " epochs")
    print("Runtime budget: " + str(RUNTIME_SECONDS) + " seconds")
    print("=" * 50)

    search_log = []
    best_val_mae = float("inf")
    best_config = None
    best_model = None
    best_test_data = None
    best_mean = None
    best_std = None
    trial_count = 0

    def target_function(config, fidelity, **kwargs):
        """
        The function DEHB calls for every configuration it wants to evaluate
        DEHB passes in the configuration and fidelity, and this function calls
        train_and_evaluate() to do the actual training. It returns the result
        to DEHB as a dictionary with the validation MAE as the fitness value (value to minimize)
        and the training time as the cost (ttime taken to evaluate this configuration)
        """
        nonlocal best_val_mae, best_config, best_model, best_test_data, best_mean, best_std, trial_count

        trial_count = trial_count + 1
        config_dict = dict(config)

        print("Trial " + str(trial_count) +
              " | Fidelity: " + str(int(fidelity)) + " epochs" +
              " | Config: lr=" + str(round(config_dict["learning_rate"], 5)) +
              " hidden=" + str(config_dict["hidden_units"]) +
              " layers=" + str(config_dict["num_layers"]))

        val_mae, cost, model, test_data, mean, std = train_and_evaluate(
            config_dict, fidelity, dataset_config, logger
        )

        print("  Val MAE: " + str(round(val_mae, 4)) + " | Cost: " + str(round(cost, 1)) + "s")

        log_entry = {
            "trial": trial_count,
            "fidelity": int(fidelity),
            "config": config_dict,
            "val_mae": val_mae,
            "cost": cost,
        }
        search_log.append(log_entry)

        with open(log_path, "w") as f:
            json.dump(search_log, f, indent=2)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = config_dict
            best_model = model
            best_test_data = test_data
            best_mean = mean
            best_std = std
            print("  *** New best: Val MAE = " + str(round(best_val_mae, 4)) + " ***")

        return {"fitness": val_mae, "cost": cost}

    dimensions = len(list(cs.values()))

    optimizer = DEHB(
        f=target_function,
        cs=cs,
        dimensions=dimensions,
        min_fidelity=MIN_FIDELITY,
        max_fidelity=MAX_FIDELITY,
        eta=3,
        n_workers=1,
        output_path=dehb_log_dir,
    )

    optimizer.run(total_cost=RUNTIME_SECONDS)

    print("")
    print("DEHB search complete for " + dataset_name)
    print("Total trials: " + str(trial_count))
    print("Best val MAE: " + str(round(best_val_mae, 4)))
    print("Best config:")
    print(json.dumps(best_config, indent=2))

    if best_model is not None:
        test_inputs, test_targets = create_windows(best_test_data, SEQ_LEN, HORIZON)
        best_model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for input_batch, target_batch in make_batches(
                test_inputs, test_targets, int(best_config["batch_size"])
            ):
                model_input = prepare_model_input(input_batch)
                b, _, n = input_batch.shape
                output = best_model(model_input).reshape(HORIZON, b, n).permute(1, 0, 2)
                all_preds.append(output.cpu().numpy())
                all_targets.append(target_batch.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        tgts = np.concatenate(all_targets, axis=0)
        test_mae, test_rmse, test_mape = compute_metrics(preds, tgts, best_mean, best_std)

        print("Test MAE:  " + str(round(test_mae, 4)))
        print("Test RMSE: " + str(round(test_rmse, 4)))
        print("Test MAPE: " + str(round(test_mape, 4)) + "%")
        print("Search log saved to " + log_path)

        return best_config, best_val_mae, test_mae, test_rmse, test_mape

    return best_config, best_val_mae, None, None, None


def main():
    config_index = 0
    while config_index < len(DATASET_CONFIGS):
        run_dehb(DATASET_CONFIGS[config_index])
        config_index = config_index + 1


if __name__ == "__main__":
    main()