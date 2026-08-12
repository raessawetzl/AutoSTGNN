import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tsl.nn.models import STCNModel

from dataloader import get_dataloaders
from metrics import compute_metrics, evaluate_per_horizon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZONS = (3, 6, 12)  # single-step-at-h, matches DCRNN/STCN/GraphWaveNet reporting convention

def log_results_csv(csv_path, dataset_name, algorithm, config, epochs, per_horizon_metrics):
    """Appends one row per horizon, in a flat shape ready for pandas pivoting later."""
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["dataset", "algorithm", "horizon", "MAE", "RMSE", "MAPE", "epochs", "config"])
        for h, m in per_horizon_metrics.items():
            writer.writerow([
                dataset_name, algorithm, h,
                f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}",
                epochs, str(config)
            ])


def train(config, dataset_name="METR-LA", epochs=50, algorithm="random_search"):

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42) # gpu reproducibiity

    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        dataset_name,
        batch_size=config["batch_size"]
    )

    sample = next(iter(train_loader))
    edge_index = sample.edge_index.to(DEVICE)
    edge_weight = sample.edge_weight.to(DEVICE)

    model = STCNModel(
        input_size=1,
        exog_size=0,
        hidden_size=config["hidden_units"],
        ff_size=config["hidden_units"],
        output_size=1,
        n_layers=config["num_layers"],
        horizon=12,
        temporal_kernel_size=2,
        spatial_kernel_size=2,
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    loss_fn = nn.L1Loss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch.x.to(DEVICE)
            y = batch.y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, edge_index, edge_weight)
            loss = loss_fn(pred[batch.mask], y[batch.mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # prevent blowup 
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch.x.to(DEVICE)
                y = batch.y.to(DEVICE)
                pred = model(x, edge_index, edge_weight)
                val_losses.append(loss_fn(pred[batch.mask], y[batch.mask]).item())

        val_mae = np.mean(val_losses)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1}/{epochs} — train MAE: {np.mean(train_losses):.4f}, val MAE: {val_mae:.4f}")

    # safety check for bad runs
    if best_state is None:
        print("  This config diverged (NaN) — skipping it")
        return float("nan")

    # reload best checkpoint before final test evaluation
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(DEVICE)
            y = batch.y.to(DEVICE)
            pred = model(x, edge_index, edge_weight)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            all_masks.append(batch.mask.cpu())   # True/1 = valid, False/0 = missing

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mask = torch.cat(all_masks).bool()

    per_horizon_metrics = evaluate_per_horizon(preds, targets, mask=mask)

    print("  --- Per-horizon test results (real units) ---")
    for h, m in per_horizon_metrics.items():
        print(f"  Horizon {h} — MAE: {m['MAE']:.4f}, RMSE: {m['RMSE']:.4f}, MAPE: {m['MAPE']:.2f}%")

    # human-readable log
    results_path = Path(__file__).resolve().parents[1] / "results" / "stcn_baseline_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(f"\n{algorithm} — {dataset_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Config: {config}\n")
        f.write(f"Epochs: {epochs}\n")
        for h, m in per_horizon_metrics.items():
            f.write(f"Horizon {h}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, MAPE={m['MAPE']:.2f}%\n")

    # paper-ready flat CSV
    csv_path = Path(__file__).resolve().parents[1] / "results" / "results.csv"
    log_results_csv(csv_path, dataset_name, algorithm, config, epochs, per_horizon_metrics)

    print(f"Results saved to {results_path} and {csv_path}")

    return best_val_mae


if __name__ == "__main__":
    config = {
        "learning_rate": 1e-3,
        "num_layers": 2,
        "hidden_units": 64,
        "dropout": 0.1,
        "batch_size": 64,
        "weight_decay": 1e-4,
    }

    print("-"*60)
    print("Baseline STCN — METR-LA")
    print("-"*60)
    train(config, dataset_name="METR-LA", epochs=15, algorithm="random_search")

    print("\n" + "-"*60)
    print("Baseline STCN — PEMS-BAY")
    print("-"*60)
    train(config, dataset_name="PEMS-BAY", epochs=15, algorithm="random_search")