"""
rs_final.py

retrains the best config found by random search, for a final fair comparison
against bohb's final retrain.

locates the random-search json log for MODEL_NAME/DATASET_NAME, picks the
trial with the lowest best_val_mae (never selecting on test metrics), and
retrains that config for FINAL_EPOCHS. writes the result to json + excel.

usage:
    python rs_final.py

config:
    model_name    - model that was searched over (stgcn, graphwavenet, agcrn, dcrnn)
    dataset_name  - dataset used (metrla, pemsbay, pems04, pems08, electricity)
    final_epochs  - epochs to retrain the winning config for
    seed          - random seed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from datetime import datetime

from utils import results_to_excel
from trainer import train

MODEL_NAME   = "stgcn"          # options: stgcn, graphwavenet, agcrn, dcrnn
DATASET_NAME = "electricity"    # options: metrla, pemsbay, pems04, pems08, electricity
FINAL_EPOCHS = 30
SEED         = 42

BASE_DIR     = Path('/content/drive/MyDrive/AutoSTGNN')
DATA_ROOT    = str(BASE_DIR / 'data')
RS_DIR       = BASE_DIR / 'Shared Results' / 'Random Search' / 'stgcn' / 'electricity'

OUT_PATH     = RS_DIR / f"rs_final_{MODEL_NAME}_{DATASET_NAME}_seed{SEED}.json"


def check_paths():
    """verify RS_DIR and the dataset dir exist, printing debug info before failing."""
    ok = True

    if not RS_DIR.is_dir():
        print(f"NOT FOUND: {RS_DIR}")
        parent = RS_DIR.parent
        while not parent.is_dir() and parent != parent.parent:
            parent = parent.parent
        print(f"  deepest existing parent: {parent}")
        print(f"  it contains: {sorted(p.name for p in parent.iterdir())}")
        ok = False

    data_dir = Path(DATA_ROOT) / DATASET_NAME
    if not data_dir.is_dir():
        print(f"NOT FOUND: {data_dir}  (tsl will try to download)")
    else:
        print(f"Data OK: {data_dir}")

    if not ok:
        raise SystemExit("Fix the paths above and re-run.")


def find_rs_log():
    """locate the random-search json log under RS_DIR, most recent if several."""
    candidates = [p for p in RS_DIR.rglob("*.json") if "rs_final" not in p.name]

    if not candidates:
        print(f"No JSON files in {RS_DIR}")
        print(f"  contents: {sorted(p.name for p in RS_DIR.rglob('*'))}")
        raise SystemExit("Point RS_DIR at the folder holding the random-search JSON.")

    candidates.sort(key=lambda p: p.stat().st_mtime)
    if len(candidates) > 1:
        print(f"Found {len(candidates)} JSON files, using most recent: {candidates[-1].name}")
    return candidates[-1]


def pick_winner(log_path):
    """return the trial with the lowest best_val_mae (never selects on test metrics)."""
    with open(log_path) as f:
        results = json.load(f)

    valid = [r for r in results if r.get("best_val_mae") is not None]
    if not valid:
        raise SystemExit(f"No trials with a recorded best_val_mae in {log_path.name}")

    winner = min(valid, key=lambda r: r["best_val_mae"])

    print(f"\nRead {log_path.name}: {len(results)} trials, {len(valid)} valid.")
    print(f"Winner: trial {winner['trial']}, val_mae = {winner['best_val_mae']:.4f}")
    if "test_mae" in winner:
        print(f"  (its search-time test_mae was {winner['test_mae']:.4f} at the "
              f"search budget — NOT comparable to BOHB's final)")
    return winner


def main():
    """find the rs winner, retrain it for FINAL_EPOCHS, and log the result."""
    check_paths()

    log_path = find_rs_log()
    winner = pick_winner(log_path)

    lr = float(winner["lr"])
    batch_size = int(winner["batch_size"])
    model_kwargs = dict(winner["model_kwargs"])

    print("\n" + "=" * 60)
    print(f"Retraining RS winner for {FINAL_EPOCHS} epochs")
    print(f"  lr={lr}  batch_size={batch_size}")
    print(f"  model_kwargs={model_kwargs}")
    print(f"  seed={SEED}")
    print("=" * 60 + "\n")

    start = time.time()

    predictor, lightning_trainer, test_results, best_model_path, best_val_mae = train(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        window=12,
        horizon=12,
        batch_size=batch_size,
        lr=lr,
        max_epochs=FINAL_EPOCHS,
        base_root=DATA_ROOT,
        model_kwargs=model_kwargs,
        seed=SEED,
    )

    duration = time.time() - start

    test_metrics = {
        k: v for k, v in test_results[0].items() if k.startswith("test_")
    }

    final_record = {
        "trial": "FINAL",
        "algorithm": "random_search",
        "seed": SEED,
        "budget_epochs": FINAL_EPOCHS,
        "epochs_run": lightning_trainer.current_epoch,
        "status": "ok",
        "source_trial": winner["trial"],
        "source_log": log_path.name,
        "lr": lr,
        "batch_size": batch_size,
        "model_kwargs": model_kwargs,
        "best_model_path": best_model_path,
        "best_val_mae": best_val_mae,
        **test_metrics,
        "trial_duration_sec": duration,
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(OUT_PATH, "w") as f:
        json.dump([final_record], f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    try:
        results_to_excel(str(OUT_PATH))
    except Exception as e:
        print(f"Excel export failed: {e}")

    print("\n" + "=" * 60)
    print(f"RS winner retrained at {FINAL_EPOCHS} epochs")
    print(f"  best_val_mae : {best_val_mae:.4f}")
    for k in ["test_mae", "test_mae_at_15", "test_mae_at_30", "test_mae_at_60"]:
        if k in test_metrics:
            print(f"  {k:16s}: {test_metrics[k]:.4f}")
    print(f"\nSaved to {OUT_PATH}")
    print("=" * 60)

    return final_record


if __name__ == "__main__":
    main()