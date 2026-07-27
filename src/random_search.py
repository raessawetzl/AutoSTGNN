import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from search_space import get_config_space



from dataloader import get_dataloaders


# ---------------------------------------------------------------------------
# Reuse everything from trainer.py
# ---------------------------------------------------------------------------
from trainer import (
    load_model,
    forward,
    inverse_transform,
    masked_mae,
    masked_rmse,
    masked_mape,
)

# ---------------------------------------------------------------------------
# Single training run — returns best val MAE
# ---------------------------------------------------------------------------
def train_one_run(base_args, config, train_loader, val_loader, mean, std, adj_mx, device):
    # merge sampled config into a copy of base_args
    args = copy.deepcopy(base_args)
    args.lr           = config["lr"]
    args.weight_decay = config["weight_decay"]
    args.rnn_units    = int(config["rnn_units"])
    args.num_layers   = int(config["num_layers"])

    if args.model == "AGCRN":
        args.embed_dim = int(config["embed_dim"])
        args.cheb_k    = int(config["cheb_k"])
    elif args.model == "DCRNN":
        args.max_diffusion_step = int(config["max_diffusion_step"])
    elif args.model == "STGCN":
        args.K       = int(config["K"])
        args.dropout = float(config["dropout"])

    model = load_model(args.model, args, adj_mx, device)
    Lk = getattr(args, "Lk", None)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae      = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss  = 0.0
        num_batches = 0
        batches_seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = forward(model, args.model, x_batch, y_batch, args,
                            teacher_forcing_ratio=1.0, Lk=Lk,
                            batches_seen=batches_seen)

            preds_real  = inverse_transform(preds,   mean, std)
            labels_real = inverse_transform(y_batch, mean, std)

            loss = masked_mae(preds_real, labels_real)
            loss.backward()

            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            total_loss   += loss.item()
            num_batches  += 1
            batches_seen += 1

        train_loss = total_loss / num_batches

        # --- validate ---
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = forward(model, args.model, x_batch, y_batch, args,
                                teacher_forcing_ratio=0.0, Lk=Lk,
                                batches_seen=None)
                all_preds.append(preds)
                all_labels.append(y_batch)

        all_preds  = torch.cat(all_preds,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        preds_real  = inverse_transform(all_preds,  mean, std)
        labels_real = inverse_transform(all_labels, mean, std)
        val_mae = masked_mae(preds_real, labels_real).item()

        print("  Epoch {:03d} | Train MAE: {:.4f} | Val MAE: {:.4f}".format(
              epoch, train_loss, val_mae), flush=True)

        if val_mae < best_val_mae:
            best_val_mae      = val_mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("  Early stopping at epoch {}".format(epoch), flush=True)
            break

    return best_val_mae


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------
def random_search(base_args, train_loader, val_loader, mean, std, adj_mx, device, n_trials):
    cs = get_config_space(base_args.model)

    results = []
    best_mae    = float("inf")
    best_config = None

    for trial in range(1, n_trials + 1):
        config = cs.sample_configuration()
        print("\n--- Trial {}/{} ---".format(trial, n_trials), flush=True)
        print("Config:", dict(config), flush=True)

        try:
            val_mae = train_one_run(
                base_args, config,
                train_loader, val_loader,
                mean, std, adj_mx, device
            )
            results.append({"config": dict(config), "val_mae": val_mae})
            print("Trial {} Val MAE: {:.4f}".format(trial, val_mae), flush=True)

            if val_mae < best_mae:
                best_mae    = val_mae
                best_config = dict(config)
                print("*** New best: {:.4f} ***".format(best_mae), flush=True)

        except Exception as e:
            print("Trial {} failed: {}".format(trial, e), flush=True)
            results.append({"config": dict(config), "val_mae": float("inf")})

    return best_config, best_mae, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--dataset",      type=str,   required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--model",        type=str,   required=True, choices=["AGCRN", "DCRNN", "STGCN"])
    parser.add_argument("--n_trials",     type=int,   default=20)
    # data
    parser.add_argument("--input_dim",    type=int,   default=1)
    parser.add_argument("--output_dim",   type=int,   default=1)
    parser.add_argument("--horizon",      type=int,   default=12)
    # fixed model defaults (overridden by ConfigSpace during search)
    parser.add_argument("--num_layers",   type=int,   default=2)
    parser.add_argument("--rnn_units",    type=int,   default=64)
    parser.add_argument("--embed_dim",    type=int,   default=10)
    parser.add_argument("--cheb_k",       type=int,   default=2)
    parser.add_argument("--max_diffusion_step", type=int, default=2)
    parser.add_argument("--K",            type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.1)
    # training — these apply to every trial
    parser.add_argument("--epochs",       type=int,   default=30)   # keep low for HPO
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--clip_grad",    type=float, default=5.0)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=64)

    args = parser.parse_args()

    NUM_SENSORS = {"PEMS04": 307, "PEMS08": 170}
    args.num_nodes = NUM_SENSORS[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Model:", args.model, flush=True)
    print("Dataset:", args.dataset, flush=True)
    print("Trials:", args.n_trials, flush=True)
    print("", flush=True)

    # load data once — shared across all trials
    train_loader, val_loader, test_loader, mean, std, adj_mx = get_dataloaders(
        args.dataset, batch_size=args.batch_size
    )
    mean = torch.tensor(mean, dtype=torch.float32).to(device)
    std  = torch.tensor(std,  dtype=torch.float32).to(device)

    best_config, best_mae, results = random_search(
        args, train_loader, val_loader, mean, std, adj_mx, device, args.n_trials
    )

    print("\n========== Random Search Complete ==========", flush=True)
    print("Best Val MAE: {:.4f}".format(best_mae), flush=True)
    print("Best Config:", best_config, flush=True)

    # save results
    import json
    results_path = "{}_{}_hpo_results.json".format(args.dataset, args.model)
    with open(results_path, "w") as f:
        json.dump({"best_config": best_config, "best_mae": best_mae, "trials": results}, f, indent=2)
    print("Results saved to:", results_path, flush=True)


if __name__ == "__main__":
    main()