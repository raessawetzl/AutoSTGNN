import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

from dataloader import get_dataloaders


# ---------------------------------------------------------------------------
# Minimal logger for DCRNN (it requires a logger object)
# ---------------------------------------------------------------------------
def get_logger():
    logger = logging.getLogger("trainer")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.WARNING)  # suppress debug/info from DCRNN internals
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Chebyshev polynomial precomputation for STGCN
# ---------------------------------------------------------------------------
def compute_chebyshev(adj_mx, K, device):
    adj = torch.FloatTensor(adj_mx).to(device)
    n = adj.shape[0]
    d = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    L = torch.eye(n).to(device) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    lambda_max = torch.linalg.eigvalsh(L).max()
    L_scaled = (2.0 * L / lambda_max) - torch.eye(n).to(device)
    Lk = [torch.eye(n).to(device), L_scaled]
    for k in range(2, K):
        Lk.append(2 * torch.mm(L_scaled, Lk[-1]) - Lk[-2])
    return torch.stack(Lk, dim=0)  # (K, N, N)


# ---------------------------------------------------------------------------
# Model registry
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
        args.Lk = None

    elif model_name == "DCRNN":
        from models.DCRNN.dcrnn_model import DCRNNModel
        logger = get_logger()
        model = DCRNNModel(
            adj_mx,
            logger,
            num_nodes            = args.num_nodes,
            input_dim            = args.input_dim,
            output_dim           = args.output_dim,
            horizon              = args.horizon,
            num_rnn_layers       = args.num_layers,
            rnn_units            = args.rnn_units,
            seq_len              = 12,
            max_diffusion_step   = args.max_diffusion_step,
            filter_type          = "laplacian",
            use_curriculum_learning = False,
        )
        args.Lk = None

    elif model_name == "STGCN":
        from models.STGCN.stgcn import STGCN
        model = STGCN(
            num_sensors  = args.num_nodes,
            num_layers   = args.num_layers,
            K            = args.K,
            horizon      = args.horizon,
            hidden_units = args.rnn_units,
            dropout      = args.dropout,
        )
        args.Lk = compute_chebyshev(adj_mx, args.K, device)

    else:
        raise ValueError("Unknown model: {}. Choose from AGCRN, DCRNN, STGCN".format(model_name))

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
# Shape helpers for DCRNN
# DCRNN input:  (seq_len, batch, num_nodes * input_dim)
# DCRNN output: (horizon, batch, num_nodes * output_dim)
# Trainer uses: (batch, T, num_nodes, dim)
# ---------------------------------------------------------------------------
def to_dcrnn_input(x):
    # x: (B, T, N, C) -> (T, B, N*C)
    B, T, N, C = x.shape
    return x.permute(1, 0, 2, 3).reshape(T, B, N * C)


def from_dcrnn_output(out, num_nodes, output_dim):
    # out: (T, B, N*C) -> (B, T, N, C)
    T, B, _ = out.shape
    return out.reshape(T, B, num_nodes, output_dim).permute(1, 0, 2, 3)


# ---------------------------------------------------------------------------
# Forward pass — handles models with different signatures
# ---------------------------------------------------------------------------
def forward(model, model_name, x, y, args, teacher_forcing_ratio=1.0, Lk=None,
            batches_seen=None):
    if model_name == "AGCRN":
        return model(x, y, teacher_forcing_ratio=teacher_forcing_ratio)

    elif model_name == "DCRNN":
        x_in = to_dcrnn_input(x)
        y_in = to_dcrnn_input(y) if y is not None else None
        out = model(x_in, y_in, batches_seen=batches_seen)
        return from_dcrnn_output(out, args.num_nodes, args.output_dim)

    elif model_name == "STGCN":
        return model(x, Lk)

    else:
        return model(x)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_one_epoch(model, model_name, loader, optimizer, mean, std,
                    clip_grad, device, args, Lk=None):
    model.train()
    total_loss   = 0.0
    num_batches  = 0
    batches_seen = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = forward(model, model_name, x_batch, y_batch, args,
                        teacher_forcing_ratio=1.0, Lk=Lk,
                        batches_seen=batches_seen)

        preds_real  = inverse_transform(preds,   mean, std)
        labels_real = inverse_transform(y_batch, mean, std)

        loss = masked_mae(preds_real, labels_real)
        loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss   += loss.item()
        num_batches  += 1
        batches_seen += 1

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, model_name, loader, mean, std, device, args, Lk=None):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = forward(model, model_name, x_batch, y_batch, args,
                            teacher_forcing_ratio=0.0, Lk=Lk,
                            batches_seen=None)
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
    parser.add_argument("--dataset",      type=str,   required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--model",        type=str,   required=True, choices=["AGCRN", "DCRNN", "STGCN"])
    # data
    parser.add_argument("--input_dim",    type=int,   default=1)
    parser.add_argument("--output_dim",   type=int,   default=1)
    parser.add_argument("--horizon",      type=int,   default=12)
    # shared model
    parser.add_argument("--num_layers",   type=int,   default=2)
    parser.add_argument("--rnn_units",    type=int,   default=64)
    # AGCRN specific
    parser.add_argument("--embed_dim",    type=int,   default=10)
    parser.add_argument("--cheb_k",       type=int,   default=2)
    # DCRNN specific
    parser.add_argument("--max_diffusion_step", type=int, default=2)
    # STGCN specific
    parser.add_argument("--K",            type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.1)
    # training
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--clip_grad",    type=float, default=5.0)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--batch_size",   type=int,   default=64)

    args = parser.parse_args()

    NUM_SENSORS = {"PEMS04": 307, "PEMS08": 170}
    args.num_nodes = NUM_SENSORS[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Model:", args.model, flush=True)
    print("Dataset:", args.dataset, flush=True)
    print("", flush=True)

    # --- data ---
    train_loader, val_loader, test_loader, mean, std, adj_mx = get_dataloaders(
        args.dataset, batch_size=args.batch_size
    )
    mean = torch.tensor(mean, dtype=torch.float32).to(device)
    std  = torch.tensor(std,  dtype=torch.float32).to(device)

    # --- model ---
    model = load_model(args.model, args, adj_mx, device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters:", num_params, flush=True)
    print("", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- training loop ---
    best_val_mae      = float("inf")
    best_epoch        = 0
    epochs_no_improve = 0
    best_model_path   = "{}_{}_{}_best.pt".format(
        args.dataset, args.model, time.strftime("%Y%m%d_%H%M%S")
    )
    best_state = None
    Lk = getattr(args, "Lk", None)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, args.model, train_loader, optimizer,
            mean, std, args.clip_grad, device, args, Lk=Lk
        )
        val_mae, val_rmse, val_mape = evaluate(
            model, args.model, val_loader, mean, std, device, args, Lk=Lk
        )

        print("Epoch {:03d} | Train MAE: {:.4f} | Val MAE: {:.4f} | "
              "Val RMSE: {:.4f} | Val MAPE: {:.2f}%".format(
              epoch, train_loss, val_mae, val_rmse, val_mape), flush=True)

        if val_mae < best_val_mae:
            best_val_mae      = val_mae
            best_epoch        = epoch
            epochs_no_improve = 0
            best_state        = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("\nEarly stopping at epoch {} "
                  "(no improvement for {} epochs)".format(epoch, args.patience), flush=True)
            break

    # --- test ---
    print("\nBest epoch: {} | Best val MAE: {:.4f}".format(best_epoch, best_val_mae), flush=True)
    print("", flush=True)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_mae, test_rmse, test_mape = evaluate(
        model, args.model, test_loader, mean, std, device, args, Lk=Lk
    )

    print("--- Test Results ({} | {}) ---".format(args.dataset, args.model), flush=True)
    print("MAE:  {:.4f}".format(test_mae), flush=True)
    print("RMSE: {:.4f}".format(test_rmse), flush=True)
    print("MAPE: {:.2f}%".format(test_mape), flush=True)


if __name__ == "__main__":
    main()