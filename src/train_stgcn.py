import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from dataloader import get_dataloaders
from models.stgcn import STGCN
from graph_utils import get_laplacian_tensor
from metrics import compute_metrics


def train(config, dataset_name="PEMS04", epochs=50):
    """
    Trains STGCN with a given config dictionary.
    Returns validation MAE — used by random search and later by BOHB.
    """

    # load data
    train_loader, val_loader, test_loader, mean, std, adj_mx = get_dataloaders(
        dataset_name,
        batch_size=config["batch_size"]
    )

    # number of sensors
    num_sensors = adj_mx.shape[0]

    # build Laplacian
    K = config["K_cheb"]
    Lk = get_laplacian_tensor(adj_mx, K)

    # build model
    model = STGCN(
        num_sensors=num_sensors,
        num_layers=config["num_layers"],
        hidden_units=config["hidden_units"],
        K=K,
        dropout=config["dropout"],
        horizon=12
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    loss_fn = nn.L1Loss()

    # training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch, Lk)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch, Lk)
                loss = loss_fn(pred, y_batch)
                val_losses.append(loss.item())

        val_mae = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{epochs} — train loss: {np.mean(train_losses):.4f}, val MAE: {val_mae:.4f}")

    # test evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch, Lk)
            all_preds.append(pred)
            all_targets.append(y_batch)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    print(f"Test MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")

    return val_mae


if __name__ == "__main__":
    # default config for a quick test run
    config = {
        "learning_rate": 1e-3,
        "num_layers": 2,
        "hidden_units": 64,
        "dropout": 0.1,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "K_cheb": 3
    }

    val_mae = train(config, dataset_name="PEMS04", epochs=5)
    print(f"\nFinal validation MAE: {val_mae:.4f}")