"""
train_best_config.py

Given a single search-results .xlsx (from random_search.py or BODE.py), extracts
the model/dataset combo and the best-scoring hyperparameter config, retrains that
config for a fixed number of epochs, and writes a one-row summary to
<results_dir>/<model>_<dataset>_<method>_final.xlsx -- same results_dir used by
run_random_search()/run_bode(), and the same base_root for input data.

Matches the real trainer.py interface as used in random_search.py:

    predictor, trainer, test_results, best_model_path, best_val_mae = train(
        dataset_name=dataset_name,
        model_name=model_name,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        base_root=base_root,
        model_kwargs=model_kwargs,
        patience=max_epochs,
    )

Usage:
    from train_best_config import run_final

    run_final(
        "agcrn_pemsbay_BODE__20260827.xlsx",
        src_dir="/content/drive/MyDrive/AutoSTGNN/src",
        max_epochs=30,
        base_root="/content/drive/MyDrive/AutoSTGNN/data",
        results_dir="/content/drive/MyDrive/AutoSTGNN/search_results",
    )
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd

FNAME_RE = re.compile(
    r"^(?:\d+_)?(?P<model>[A-Za-z0-9]+)_(?P<dataset>[A-Za-z0-9]+)_"
    r"(?P<method>BODE|RS)_+(?P<date>\d+)\.xlsx$",
    re.IGNORECASE,
)

# Mirrors search_space.py's get_search_space(): which hyperparams are
# integer/categorical-integer vs boolean vs float, per model. Needed because
# pd.read_excel() returns numpy floats for everything, and things like
# DataLoader(batch_size=...) reject numpy types / floats where a plain Python
# int or bool is required.
INT_PARAMS_COMMON = {"batch_size", "hidden_size"}
BOOL_PARAMS = {"learned_adjacency"}

INT_PARAMS_BY_MODEL = {
    "graphwavenet": {"ff_size", "n_layers", "emb_size"},
    "dcrnn": {"ff_size", "kernel_size", "n_layers"},
    "stgcn": {"ff_size", "n_layers", "temporal_kernel_size", "spatial_kernel_size"},
    "agcrn": {"n_layers", "emb_size"},
}


def coerce_value(model_name, key, value):
    if key in BOOL_PARAMS:
        # Excel may round-trip True/False as 1.0/0.0 or as strings.
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1")
        return bool(round(float(value)))
    if key in INT_PARAMS_COMMON or key in INT_PARAMS_BY_MODEL.get(model_name, set()):
        return int(round(float(value)))
    return float(value)


def parse_model_dataset_method(path: Path, model_override: str, dataset_override: str, method_override: str = None):
    if model_override and dataset_override:
        return model_override.lower(), dataset_override.lower(), (method_override or "UNKNOWN").upper()
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Couldn't parse model/dataset/method from filename '{path.name}'. "
            f"Expected <model>_<dataset>_<BODE|RS>_<date>.xlsx, or pass "
            f"--model/--dataset/--method explicitly."
        )
    return m.group("model").lower(), m.group("dataset").lower(), m.group("method").upper()


def value_column(df: pd.DataFrame) -> str:
    if "val_mae" in df.columns:
        return "val_mae"
    if "best_val_mae" in df.columns:
        return "best_val_mae"
    raise KeyError("Neither 'val_mae' nor 'best_val_mae' found in columns")


def extract_best_config(df: pd.DataFrame, model_name: str) -> dict:
    col = value_column(df)
    best_idx = df[col].astype(float).idxmin()
    row = df.loc[best_idx]

    lr = float(row["lr"])
    batch_size = int(round(float(row["batch_size"])))
    model_kwargs = {
        c[3:]: coerce_value(model_name, c[3:], row[c])
        for c in df.columns
        if c.startswith("kw_")
    }

    return {
        "trial": row.get("trial", best_idx),
        "best_val_mae_in_search": row[col],
        "lr": lr,
        "batch_size": batch_size,
        "model_kwargs": model_kwargs,
    }


def call_train(model_name, dataset_name, lr, batch_size, model_kwargs, max_epochs, window, horizon, base_root):
    """
    Thin wrapper around src/trainer.py's train(). Adjust here if the real
    signature changes -- everything else in this script is agnostic to it.
    """
    from trainer import train  # imported here so src_dir can be put on sys.path first

    return train(
        dataset_name=dataset_name,
        model_name=model_name,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        base_root=base_root,
        model_kwargs=model_kwargs,
        patience=max_epochs,
    )


def run_final(
    xlsx_path,
    model_name=None,
    dataset_name=None,
    method_name=None,
    max_epochs=30,
    window=12,
    horizon=12,
    base_root="./data",
    results_dir="./search_results",
    src_dir=".",
):
    """
    Extract the best config from xlsx_path, retrain it for max_epochs, and write
    <results_dir>/<model>_<dataset>_<method>_final.xlsx.

    base_root and results_dir mirror run_random_search()'s params exactly --
    same input data root, same output directory as the search results live in.

    Returns the output row as a dict.
    """
    xlsx_path = Path(xlsx_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(src_dir).resolve()))

    if not xlsx_path.exists():
        raise FileNotFoundError(f"file not found: {xlsx_path}")

    model, dataset, method = parse_model_dataset_method(xlsx_path, model_name, dataset_name, method_name)
    df = pd.read_excel(xlsx_path)
    best = extract_best_config(df, model)

    print(f"Model: {model}  Dataset: {dataset}  Method: {method}")
    print(f"Best trial: {best['trial']}  (search val_mae={best['best_val_mae_in_search']:.4f})")
    print(f"lr={best['lr']}, batch_size={best['batch_size']}, model_kwargs={best['model_kwargs']}")

    trial_start = time.time()
    predictor, trainer, test_results, best_model_path, best_val_mae = call_train(
        model_name=model,
        dataset_name=dataset,
        lr=best["lr"],
        batch_size=best["batch_size"],
        model_kwargs=best["model_kwargs"],
        max_epochs=max_epochs,
        window=window,
        horizon=horizon,
        base_root=base_root,
    )
    duration_sec = time.time() - trial_start

    test_results_dict = test_results[0]
    test_metrics = {k: v for k, v in test_results_dict.items() if k.startswith("test_")}

    out_row = {
        "model": model,
        "dataset": dataset,
        "method": method,
        "max_epochs": max_epochs,
        "source_file": xlsx_path.name,
        "source_best_trial": best["trial"],
        "search_val_mae": best["best_val_mae_in_search"],
        "lr": best["lr"],
        "batch_size": best["batch_size"],
        "model_kwargs": best["model_kwargs"],
        "best_model_path": best_model_path,
        "best_val_mae": best_val_mae,
        **test_metrics,
        "trial_duration_sec": duration_sec,
    }

    # Prefer the project's own JSON->xlsx flattener for consistent kw_* columns,
    # matching the format of the RS/BODE result files. Falls back to a direct
    # pandas write if utils.results_to_excel isn't importable/available.
    out_path = results_dir / f"{model}_{dataset}_{method}_final.xlsx"
    try:
        import json
        from utils import results_to_excel

        json_path = results_dir / f"{model}_{dataset}_{method}_final.json"
        with open(json_path, "w") as f:
            json.dump([out_row], f, indent=2, default=str)
        results_to_excel(str(json_path))
    except Exception as e:
        print(f"results_to_excel unavailable/failed ({e}); writing xlsx directly instead.")
        flat_row = {**{k: v for k, v in out_row.items() if k != "model_kwargs"},
                    **{f"kw_{k}": v for k, v in best["model_kwargs"].items()}}
        pd.DataFrame([flat_row]).to_excel(out_path, index=False)

    print(f"\nWrote {out_path}")
    return out_row