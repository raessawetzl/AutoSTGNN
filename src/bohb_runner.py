import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import wandb
from pathlib import Path

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from ConfigSpace import ConfigurationSpace

from dataloader import get_dataloaders
from models.stgcn import STGCN
from graph_utils import get_laplacian_tensor
from metrics import compute_metrics
from search_space import get_stgcn_config_space

# settings

DATASET_NAME = "PEMS-BAY"       # "PEMS-BAY" or "METR-LA"
MODEL_NAME   = "STGCN"        # used for logging and output paths
N_TRIALS     = 50             # total BOHB configurations to evaluate
EPOCHS       = 50             # epochs per trial (SMAC controls budget via fidelity)
MIN_BUDGET   = 5              # minimum epochs HyperBand allocates to a trial
MAX_BUDGET   = EPOCHS         # maximum epochs HyperBand allocates to a trial
SEED         = 42
OUTPUT_DIR   = Path(__file__).resolve().parent / "bohb_results"
WANDB_PROJECT = "AutoSTGNN-BOHB"

# device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# pre-load data and adjacency matrix  — shared across all trials

_train_loader_cache = {}   # keyed by batch_size
_val_loader   = None
_test_loader  = None
_mean         = None
_std          = None
_adj_mx       = None
_Lk_cache     = {}         # keyed by K_cheb

def _ensure_data_loaded(batch_size: int):
    """Loads data splits once per batch_size value encountered."""
    global _val_loader, _test_loader, _mean, _std, _adj_mx

    if _adj_mx is None:
        # load with a dummy batch size to get val/test loaders and adj
        _, _val_loader, _test_loader, _mean, _std, _adj_mx = get_dataloaders(
            DATASET_NAME, batch_size=64
        )

    if batch_size not in _train_loader_cache:
        train_loader, _, _, _, _, _ = get_dataloaders(
            DATASET_NAME, batch_size=batch_size
        )
        _train_loader_cache[batch_size] = train_loader


def _get_lk(K: int) -> torch.Tensor:
    """Builds (and caches) the Chebyshev Laplacian for a given K."""
    if K not in _Lk_cache:
        _Lk_cache[K] = get_laplacian_tensor(_adj_mx, K).to(DEVICE)
    return _Lk_cache[K]


# target function — called by SMAC for every configuration it wants to try

def train_stgcn(config, seed: int = SEED, budget: int = EPOCHS) -> float:
    """
    Trains STGCN with the given hyperparameter configuration for `budget`
    epochs. Returns validation MAE — SMAC minimises this.

    Parameters
    ----------
    config : ConfigSpace.Configuration
        Hyperparameter configuration sampled by BOHB.
    seed : int
        Random seed for this trial (passed in by SMAC).
    budget : int
        Number of training epochs allocated by HyperBand for this trial.

    Returns
    -------
    float
        Validation MAE on the current dataset.
    """

    # reproducibility for this trial
    torch.manual_seed(seed)
    np.random.seed(seed)

    # unpack hyperparameters from config
    lr           = float(config["learning_rate"])
    num_layers   = int(config["num_layers"])
    hidden_units = int(config["hidden_units"])
    dropout      = float(config["dropout"])
    batch_size   = int(config["batch_size"])
    weight_decay = float(config["weight_decay"])
    K            = int(config["K_cheb"])

    # ensure data is loaded for this batch_size
    _ensure_data_loaded(batch_size)
    train_loader = _train_loader_cache[batch_size]
    Lk = _get_lk(K)

    num_sensors = _adj_mx.shape[0]

    # build model
    model = STGCN(
        num_sensors=num_sensors,
        num_layers=num_layers,
        hidden_units=hidden_units,
        K=K,
        dropout=dropout,
        horizon=12,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.L1Loss()

    # W&B run for this trial (set mode="disabled" to turn off)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{MODEL_NAME}_{DATASET_NAME}_trial",
        config={
            "learning_rate": lr,
            "num_layers": num_layers,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "K_cheb": K,
            "budget_epochs": budget,
            "seed": seed,
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,   
        },
        reinit="finish_previous",
        mode="offline",
    )

    # training loop
    best_val_mae = float("inf")

    for epoch in range(int(budget)):
        # train
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch, Lk)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validate 
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in _val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred = model(x_batch, Lk)
                loss = loss_fn(pred, y_batch)
                val_losses.append(loss.item())

        train_mae = np.mean(train_losses)
        val_mae   = np.mean(val_losses)

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        wandb.log({
            "epoch":     epoch + 1,
            "train_mae": train_mae,
            "val_mae":   val_mae,
        })

        print(
            f"  Epoch {epoch+1:>3}/{int(budget)} — "
            f"train MAE: {train_mae:.4f}, val MAE: {val_mae:.4f}"
        )

    run.finish()

    # SMAC minimises the return value — return best val MAE seen this trial
    return best_val_mae


# BOHB setup and run

def run_bohb():
    """Configures and runs BOHB via SMAC3. Saves the best config to disk."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cs: ConfigurationSpace = get_stgcn_config_space()

    scenario = Scenario(
        configspace=cs,
        name=f"BOHB_{MODEL_NAME}_{DATASET_NAME}",
        output_directory=OUTPUT_DIR,
        deterministic=True,       # one seed per config — set False for noisy objectives
        n_trials=N_TRIALS,
        seed=SEED,
        # hyperBand budget is in terms of the `budget` argument to train_stgcn
        min_budget=MIN_BUDGET,
        max_budget=MAX_BUDGET,
    )

    # hyperBand intensifier — this is what makes it BOHB (Bayesian + HyperBand)
    intensifier = Hyperband(
        scenario,
        incumbent_selection="highest_budget",  # compare configs at max budget
    )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train_stgcn,
        intensifier=intensifier,
        overwrite=True,
    )

    print("\n" + "="*60)
    print(f"Starting BOHB — {N_TRIALS} trials on {MODEL_NAME} / {DATASET_NAME}")
    print(f"Budget per trial: {MIN_BUDGET}–{MAX_BUDGET} epochs")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")

    incumbent = smac.optimize()

    # final evaluation of the best configuration on the test set

    print("\n" + "="*60)
    print("BOHB complete. Evaluating best configuration on test set...")
    print("="*60)
    print("Best config found:")
    for key, val in incumbent.items():
        print(f"  {key}: {val}")

    # retrain from scratch on train+val with best config at full budget
    _ensure_data_loaded(incumbent["batch_size"])
    Lk = _get_lk(incumbent["K_cheb"])
    num_sensors = _adj_mx.shape[0]

    model = STGCN(
        num_sensors=num_sensors,
        num_layers=incumbent["num_layers"],
        hidden_units=incumbent["hidden_units"],
        K=incumbent["K_cheb"],
        dropout=incumbent["dropout"],
        horizon=12,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=incumbent["learning_rate"],
        weight_decay=incumbent["weight_decay"],
    )
    loss_fn = nn.L1Loss()

    train_loader = _train_loader_cache[incumbent["batch_size"]]

    torch.manual_seed(SEED)
    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch, Lk)
            loss_fn(pred, y_batch).backward()
            optimizer.step()

    # test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in _test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(x_batch, Lk)
            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)

    print(f"\nTest results with best BOHB config:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # save results summary to disk
    results_path = OUTPUT_DIR / f"bohb_{MODEL_NAME}_{DATASET_NAME}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"BOHB Results — {MODEL_NAME} on {DATASET_NAME}\n")
        f.write("="*50 + "\n\n")
        f.write("Best hyperparameter configuration:\n")
        for key, val in incumbent.items():
            f.write(f"  {key}: {val}\n")
        f.write(f"\nTest MAE:  {metrics['MAE']:.4f}\n")
        f.write(f"Test RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"Test MAPE: {metrics['MAPE']:.2f}%\n")
        f.write(f"\nN trials: {N_TRIALS}\n")
        f.write(f"Epochs per trial (max budget): {MAX_BUDGET}\n")
        f.write(f"Min budget: {MIN_BUDGET}\n")
        f.write(f"Seed: {SEED}\n")

    print(f"\nResults saved to {results_path}")

    return incumbent, metrics

# entry point

if __name__ == "__main__":
    # initialise W&B (will prompt login if not already authenticated)
    wandb.login()
    run_bohb()
