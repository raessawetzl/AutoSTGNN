import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from dataloader import get_dataloaders

# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------
def load_model(model_name, args, adj_mx, device):
    if model_name == "AGCRN":
        from models.AGCRN.AGCRN import AGCRN
        model_args = argparse.Namespace(
            num_nodes     = args.num_nodes,
            input_dim     = args.input_dim,
            output_dim    = args.output_dim,
            embed_dim     = args.embed_dim,
            cheb_k        = args.cheb_k,
            horizon       = args.horizon,
            num_layers    = args.num_layers,
            rnn_units     = args.rnn_units,
            default_graph = True,
        )
        model = AGCRN(model_args)

    elif model_name == "DCRNN":
        from models.DCRNN.dcrnn_model import DCRNNModel
        model = DCRNNModel(
            num_nodes  = args.num_nodes,
            input_dim  = args.input_dim,
            output_dim = args.output_dim,
            horizon    = args.horizon,
            num_layers = args.num_layers,
            adj_mx     = adj_mx,
        )

    elif model_name == "STGCN":
        from models.STGCN.stgcn import STGCN
        model = STGCN(
            num_nodes  = args.num_nodes,
            input_dim  = args.input_dim,
            output_dim = args.output_dim,
            horizon    = args.horizon,
            adj_mx     = adj_mx,
        )

    else:
        raise ValueError("Unknown model: {}. Choose from AGCRN, DCRNN, STGCN".format(model_name))

    # initialise all parameters — critical for numerical stability across
    # PyTorch versions (this is what the original AGCRN repo does in Run.py)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model.to(device)


# ---------------------------------------------------------------------------
# Metrics
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
    mask = labels != mask_value
    mape = torch.abs((preds - labels) / (labels + 1e-8))
    return 100.0 * (mape * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------
# Forward pass — handles models with different signatures
# AGCRN: model(x, y, teacher_forcing_ratio)
# DCRNN: model(x, y, teacher_forcing_ratio)
# STGCN: model(x)
# ---------------------------------------------------------------------------
def forward(model, model_name, x, y, teacher_forcing_ratio=1.0):
    if model_name in ("AGCRN", "DCRNN"):
        return model(x, y, teacher_forcing_ratio=teacher_forcing_ratio)
    else:
        return model(x)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_one_epoch(model, model_name, loader, optimizer, mean, std, clip_grad, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = forward(model, model_name, x_batch, y_batch, teacher_forcing_ratio=1.0)

        preds_real  = inverse_transform(preds,   mean, std)
        labels_real = inverse_transform(y_batch, mean, std)

        loss = masked_mae(preds_real, labels_real)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, model_name, loader, mean, std, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = forward(model, model_name, x_batch, y_batch, teacher_forcing_ratio=0.0)
            all_preds.append(preds)
            all_labels.append(y_batch)

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds_real  = inverse_transform(all_preds,  mean, std)
    labels_real = inverse_transform(all_labels, mean, std)

    mae  = masked_mae(preds_real,  labels_real).item()
    rmse = masked_rmse(preds_real, labels_real).item()
    mape = masked_mape(preds_real, labels_real).item()

    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--dataset",    type=str, required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--model",      type=str, required=True, choices=["AGCRN", "DCRNN", "STGCN"])
    # data
    parser.add_argument("--input_dim",  type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--horizon",    type=int, default=12)
    # shared model
    parser.add_argument("--num_layers", type=int, default=2)
    # AGCRN specific
    parser.add_argument("--embed_dim",  type=int, default=10)
    parser.add_argument("--cheb_k",     type=int, default=2)
    parser.add_argument("--rnn_units",  type=int, default=64)
    # training
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--clip_grad",  type=float, default=5.0)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=64)

    args = parser.parse_args()

    NUM_SENSORS = {"PEMS04": 307, "PEMS08": 170}
    args.num_nodes = NUM_SENSORS[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("")

    # --- data ---
    train_loader, val_loader, test_loader, mean, std, adj_mx = get_dataloaders(
        args.dataset, batch_size=args.batch_size
    )
    mean = torch.tensor(mean, dtype=torch.float32).to(device)
    std  = torch.tensor(std,  dtype=torch.float32).to(device)

    # --- model ---
    model = load_model(args.model, args, adj_mx, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters:", num_params)
    print("")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- training loop ---
    best_val_mae  = float("inf")
    best_epoch    = 0
    epochs_no_improve = 0
    best_model_path = "{}_{}_{}_best.pt".format(args.dataset, args.model, 
                        time.strftime("%Y%m%d_%H%M%S"))
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, args.model, train_loader, optimizer,
            mean, std, args.clip_grad, device
        )
        val_mae, val_rmse, val_mape = evaluate(
            model, args.model, val_loader, mean, std, device
        )

        print("Epoch {:03d} | Train MAE: {:.4f} | Val MAE: {:.4f} | "
              "Val RMSE: {:.4f} | Val MAPE: {:.2f}%".format(
              epoch, train_loss, val_mae, val_rmse, val_mape))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("\nEarly stopping at epoch {} "
                  "(no improvement for {} epochs)".format(epoch, args.patience))
            break

    # --- test ---
    print("\nBest epoch: {} | Best val MAE: {:.4f}".format(best_epoch, best_val_mae))
    print("")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_mae, test_rmse, test_mape = evaluate(
        model, args.model, test_loader, mean, std, device
    )

    print("--- Test Results ({} | {}) ---".format(args.dataset, args.model))
    print("MAE:  {:.4f}".format(test_mae))
    print("RMSE: {:.4f}".format(test_rmse))
    print("MAPE: {:.2f}%".format(test_mape))


if __name__ == "__main__":
    main()