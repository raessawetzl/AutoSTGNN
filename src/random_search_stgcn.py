import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tsl.nn.models import STCNModel

from search_space import get_stgcn_config_space
from dataloader import get_dataloaders
from graph_utils import adj_to_edge_index
from metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def train(config_dict, dataset_name="METR-LA", epochs=10):
    torch.cuda.empty_cache()  # memory efficient 
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    lr           = float(config_dict["learning_rate"])
    num_layers   = int(config_dict["num_layers"])
    hidden_units = int(config_dict["hidden_units"])
    dropout      = float(config_dict["dropout"])
    batch_size   = int(config_dict["batch_size"])
    weight_decay = float(config_dict["weight_decay"])

    train_loader, val_loader, test_loader, _, _, adj_mx = get_dataloaders(
        dataset_name, batch_size=batch_size
    )
    edge_index, edge_weight = adj_to_edge_index(adj_mx)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

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

    best_val_mae = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch, edge_index, edge_weight)

            #debug
            x_batch, y_batch = next(iter(train_loader))
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(x_batch, edge_index, edge_weight)
            print("pred shape:", pred.shape, "target shape:", y_batch.shape)
            print("pred mean/std:", pred.mean().item(), pred.std().item())
            print("target mean/std:", y_batch.mean().item(), y_batch.std().item())

            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(x_batch, edge_index, edge_weight)
                val_losses.append(loss_fn(pred, y_batch).item())

        val_mae = np.mean(val_losses)
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        print(f"  Epoch {epoch+1}/{epochs} — train MAE: {np.mean(train_losses):.4f}, val MAE: {val_mae:.4f}")

    # test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(x_batch, edge_index, edge_weight)
            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    print(f"  Test MAE:  {metrics['MAE']:.4f}")
    print(f"  Test RMSE: {metrics['RMSE']:.4f}")
    print(f"  Test MAPE: {metrics['MAPE']:.2f}%")

    del model
    torch.cuda.empty_cache()

    return best_val_mae


def random_search(n_configs=20, epochs=10, dataset_name="METR-LA"):
    cs = get_stgcn_config_space()
    results = []

    for i in range(n_configs):
        config = cs.sample_configuration()
        config_dict = {k: int(v) if isinstance(v, np.integer) else v for k, v in dict(config).items()}

        print(f"\nConfig {i+1}/{n_configs}: {config_dict}")
        val_mae = train(config_dict, dataset_name=dataset_name, epochs=epochs)
        results.append((val_mae, config_dict))
        print(f"Val MAE: {val_mae:.4f}")

    results.sort(key=lambda x: x[0])

    print("\n--- Top 5 configs ---")
    for val_mae, cfg in results[:5]:
        print(f"Val MAE: {val_mae:.4f} | {cfg}")

    return results


if __name__ == "__main__":
    print("-"*60)
    print("Random Search on METR-LA")
    print("-"*60)
    results_metrla = random_search(n_configs=5, epochs=5, dataset_name="METR-LA")

    # print("\n" + "-"*60)
    # print("Random Search on PEMS-BAY")
    # print("-"*60)
    # results_pemsbay = random_search(n_configs=20, epochs=20, dataset_name="PEMS-BAY")

    results_path = Path(__file__).resolve().parent / "rs_metrla_results.txt"
    with open(results_path, "w") as f:

        f.write("Random Search Results — METR-LA\n")
        f.write("="*50 + "\n")
        for val_mae, cfg in results_metrla[:5]:
            f.write(f"Val MAE: {val_mae:.4f} | {cfg}\n")

        # f.write("\nRandom Search Results — PEMS-BAY\n")
        # f.write("="*50 + "\n")
        # for val_mae, cfg in results_pemsbay[:5]:
        #     f.write(f"Val MAE: {val_mae:.4f} | {cfg}\n")
    print(f"Results saved to {results_path}")