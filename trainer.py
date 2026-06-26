import argparse
import numpy as np
import torch
import torch.nn as nn

from dataloader import get_dataloaders
from AGCRN import AGCRN

# ---------------------------------------------------------------------------
# Hyperparameters
# These are the standard values used in the original AGCRN paper on PEMS04/08
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "input_dim":   1,    # flow only
    "hidden_dim":  64,
    "output_dim":  1,    # predicting flow only
    "embed_dim":   10,   # node embedding dimension
    "cheb_k":      2,    # Chebyshev polynomial order
    "horizon":     12,   # steps to predict
    "num_layers":  2,
}

TRAIN_CONFIG = {
    "epochs":       100,
    "lr":           0.001,
    "weight_decay": 0.0001,
    "clip_grad":    5.0,   # gradient clipping max norm
    "patience":     15,    # early stopping: stop if val loss doesn't improve for this many epochs
}

# num_sensors per dataset — needed to instantiate the model
NUM_SENSORS = {
    "PEMS04": 307,
    "PEMS08": 170,
}


# ---------------------------------------------------------------------------
# Loss and metrics
# All computed in real (de-normalized) units, masking out true zeros
# since MAE on a true-zero label is meaningless for sensor-gap periods
# ---------------------------------------------------------------------------

def inverse_transform(tensor, mean, std):
    return tensor * std + mean


def masked_mae(preds, labels, mask_value=0.0):
    mask = labels != mask_value
    mae = torch.abs(preds - labels)
    return (mae * mask).sum() / mask.sum()


def masked_rmse(preds, labels, mask_value=0.0):
    mask = labels != mask_value
    mse = (preds - labels) ** 2
    return torch.sqrt((mse * mask).sum() / mask.sum())


def masked_mape(preds, labels, mask_value=0.0):
    # small epsilon on denominator to avoid division near zero even after masking
    mask = labels != mask_value
    mape = torch.abs((preds - labels) / (labels + 1e-8))
    return 100.0 * (mape * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, mean, std, clip_grad, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)   # (B, T, N, 1)
        y_batch = y_batch.to(device)   # (B, T, N, 1)

        optimizer.zero_grad()
        preds = model(x_batch)         # (B, horizon, N, 1)

        # de-normalize both before computing loss so the loss is in
        # real units (vehicles/5-min) and comparable across runs
        preds_real = inverse_transform(preds, mean, std)
        labels_real = inverse_transform(y_batch, mean, std)

        loss = masked_mae(preds_real, labels_real)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# One epoch of validation — returns loss + full metrics
# ---------------------------------------------------------------------------

def evaluate(model, loader, mean, std, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            all_preds.append(preds)
            all_labels.append(y_batch)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds_real = inverse_transform(all_preds, mean, std)
    labels_real = inverse_transform(all_labels, mean, std)

    mae  = masked_mae(preds_real, labels_real).item()
    rmse = masked_rmse(preds_real, labels_real).item()
    mape = masked_mape(preds_real, labels_real).item()

    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PEMS04", "PEMS08"],
        help="Which dataset to train on"
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    num_node = NUM_SENSORS[dataset_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    print("")

    # --- data ---
    train_loader, val_loader, test_loader, mean, std, adj_mx = get_dataloaders(dataset_name)
    # adj_mx is loaded but not used by AGCRN — its adjacency is learned
    # internally via node embeddings. Pass it to DCRNN/STGCN training
    # scripts when you write those.
    mean = torch.tensor(mean, dtype=torch.float32).to(device)
    std  = torch.tensor(std,  dtype=torch.float32).to(device)

    # --- model ---
    model = AGCRN(
        num_node   = num_node,
        input_dim  = MODEL_CONFIG["input_dim"],
        hidden_dim = MODEL_CONFIG["hidden_dim"],
        output_dim = MODEL_CONFIG["output_dim"],
        embed_dim  = MODEL_CONFIG["embed_dim"],
        cheb_k     = MODEL_CONFIG["cheb_k"],
        horizon    = MODEL_CONFIG["horizon"],
        num_layers = MODEL_CONFIG["num_layers"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters: " + str(num_params))
    print("")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # --- training loop with early stopping ---
    best_val_mae = float("inf")
    best_epoch   = 0
    epochs_since_improvement = 0
    best_model_path = dataset_name + "_best_model.pt"

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            mean, std, TRAIN_CONFIG["clip_grad"], device
        )

        val_mae, val_rmse, val_mape = evaluate(model, val_loader, mean, std, device)

        print(
            "Epoch {:03d} | Train MAE: {:.4f} | Val MAE: {:.4f} | "
            "Val RMSE: {:.4f} | Val MAPE: {:.2f}%".format(
                epoch, train_loss, val_mae, val_rmse, val_mape
            )
        )

        # save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_since_improvement += 1

        # early stopping
        if epochs_since_improvement >= TRAIN_CONFIG["patience"]:
            print("")
            print("Early stopping at epoch " + str(epoch) +
                  " (no improvement for " + str(TRAIN_CONFIG["patience"]) + " epochs)")
            break

    # --- final test evaluation using best checkpoint ---
    print("")
    print("Best epoch: " + str(best_epoch) + " | Best val MAE: {:.4f}".format(best_val_mae))
    print("")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_mae, test_rmse, test_mape = evaluate(model, test_loader, mean, std, device)

    print("--- Test Results (" + dataset_name + ") ---")
    print("MAE:  {:.4f}".format(test_mae))
    print("RMSE: {:.4f}".format(test_rmse))
    print("MAPE: {:.2f}%".format(test_mape))


if __name__ == "__main__":
    main()