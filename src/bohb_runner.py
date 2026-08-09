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
from tsl.nn.models import STCNModel
from graph_utils import adj_to_edge_index
from metrics import compute_metrics
from search_space import get_stgcn_config_space

# settings

MODEL_NAME   = "STGCN"
N_TRIALS     = 50
EPOCHS       = 50
MIN_BUDGET   = 5
MAX_BUDGET   = EPOCHS
SEED         = 42
OUTPUT_DIR   = Path(__file__).resolve().parent / "bohb_results"
WANDB_PROJECT = "AutoSTGNN-BOHB"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# global caches — reset between dataset runs

_train_loader_cache = {}
_val_loader   = None
_test_loader  = None
_mean         = None
_std          = None
_adj_mx       = None
_edge_index   = None
_edge_weight  = None
_dataset_name = None


def _reset_cache():
    """Clears all cached data between dataset runs."""
    global _train_loader_cache, _val_loader, _test_loader
    global _mean, _std, _adj_mx, _edge_index, _edge_weight
    _train_loader_cache = {}
    _val_loader   = None
    _test_loader  = None
    _mean         = None
    _std          = None
    _adj_mx       = None
    _edge_index   = None
    _edge_weight  = None


def _ensure_data_loaded(batch_size: int):
    """Loads data splits once per batch_size value encountered."""
    global _val_loader, _test_loader, _mean, _std, _adj_mx

    if _adj_mx is None:
        _, _val_loader, _test_loader, _mean, _std, _adj_mx = get_dataloaders(
            _dataset_name, batch_size=64
        )
        global _edge_index, _edge_weight
        _edge_index, _edge_weight = adj_to_edge_index(_adj_mx)
        _edge_index = _edge_index.to(DEVICE)
        _edge_weight = _edge_weight.to(DEVICE)

    if batch_size not in _train_loader_cache:
        train_loader, _, _, _, _, _ = get_dataloaders(
            _dataset_name, batch_size=batch_size
        )
        _train_loader_cache[batch_size] = train_loader


def train_stgcn(config, seed: int = SEED, budget: int = EPOCHS) -> float:
    """
    Trains STCNModel with the given hyperparameter configuration for `budget`
    epochs. Returns validation MAE — SMAC minimises this.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr           = float(config["learning_rate"])
    num_layers   = int(config["num_layers"])
    hidden_units = int(config["hidden_units"])
    dropout      = float(config["dropout"])
    batch_size   = int(config["batch_size"])
    weight_decay = float(config["weight_decay"])

    _ensure_data_loaded(batch_size)
    train_loader = _train_loader_cache[batch_size]

    model = STCNModel(
        input_size=1,
        exog_size=0,
        hidden_size=hidden_units,
        ff_size=hidden_units,
        output_size=1,
        n_layers=num_layers,
        horizon=12,
        temporal_kernel_size=2,
        spatial_kernel_size=2,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{MODEL_NAME}_{_dataset_name}_trial",
        config={
            "learning_rate": lr,
            "num_layers": num_layers,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "budget_epochs": budget,
            "seed": seed,
            "dataset": _dataset_name,
            "model": MODEL_NAME,
        },
        reinit="finish_previous",
        mode="offline",
    )

    best_val_mae = float("inf")

    for epoch in range(int(budget)):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch, _edge_index, _edge_weight)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in _val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                pred = model(x_batch, _edge_index, _edge_weight)
                loss = loss_fn(pred, y_batch)
                val_losses.append(loss.item())

        train_mae = np.mean(train_losses)
        val_mae   = np.mean(val_losses)

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        wandb.log({"epoch": epoch + 1, "train_mae": train_mae, "val_mae": val_mae})

        print(
            f"  Epoch {epoch+1:>3}/{int(budget)} — "
            f"train MAE: {train_mae:.4f}, val MAE: {val_mae:.4f}"
        )

    run.finish()

    # memory efficiency
    del model
    torch.cuda.empty_cache()

    return best_val_mae


def run_bohb(dataset_name: str):
    """Runs BOHB for a single dataset. Saves results to disk."""
    global _dataset_name
    _dataset_name = dataset_name

    _reset_cache()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cs: ConfigurationSpace = get_stgcn_config_space()

    scenario = Scenario(
        configspace=cs,
        name=f"BOHB_{MODEL_NAME}_{dataset_name}",
        output_directory=OUTPUT_DIR,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=SEED,
        min_budget=MIN_BUDGET,
        max_budget=MAX_BUDGET,
    )

    intensifier = Hyperband(
        scenario,
        incumbent_selection="highest_budget",
    )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train_stgcn,
        intensifier=intensifier,
        overwrite=True,
    )

    print("\n" + "="*60)
    print(f"Starting BOHB — {N_TRIALS} trials on {MODEL_NAME} / {dataset_name}")
    print(f"Budget per trial: {MIN_BUDGET}–{MAX_BUDGET} epochs")
    print(f"Results will be saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")

    incumbent = smac.optimize()

    print("\n" + "="*60)
    print(f"BOHB complete for {dataset_name}. Evaluating best config on test set...")
    print("="*60)
    print("Best config found:")
    for key, val in incumbent.items():
        print(f"  {key}: {val}")

    # retrain from scratch with best config at full budget
    _ensure_data_loaded(incumbent["batch_size"])

    model = STCNModel(
        input_size=1,
        exog_size=0,
        hidden_size=incumbent["hidden_units"],
        ff_size=incumbent["hidden_units"],
        output_size=1,
        n_layers=incumbent["num_layers"],
        horizon=12,
        temporal_kernel_size=2,
        spatial_kernel_size=2,
        dropout=incumbent["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=incumbent["learning_rate"],
        weight_decay=incumbent["weight_decay"],
    )
    loss_fn = nn.L1Loss()
    train_loader = _train_loader_cache[incumbent["batch_size"]]

    torch.manual_seed(SEED)
    print(f"\nRetraining best config for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch, _edge_index, _edge_weight)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"  Retrain epoch {epoch+1}/{EPOCHS} — train MAE: {np.mean(train_losses):.4f}")

    # test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in _test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(x_batch, _edge_index, _edge_weight)
            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)

    print(f"\nTest results with best BOHB config on {dataset_name}:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")

    results_path = OUTPUT_DIR / f"bohb_{MODEL_NAME}_{dataset_name}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"BOHB Results — {MODEL_NAME} on {dataset_name}\n")
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

    print(f"Results saved to {results_path}")
    return incumbent, metrics


if __name__ == "__main__":
    wandb.login()

    print("\n" + "="*60)
    print("BOHB RUN 1: METR-LA")
    print("="*60)
    run_bohb("METR-LA")

    # print("\n" + "="*60)
    # print("BOHB RUN 2: PEMS-BAY")
    # print("="*60)
    # run_bohb("PEMS-BAY")