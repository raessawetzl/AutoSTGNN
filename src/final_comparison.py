"""
final_comparison.py

Scans search_results/ for files named:
    <model>_<dataset>_<method>_<date>.xlsx

Groups them by (model, dataset), and within each group finds every method's
best config by val_mae (falling back to best_val_mae if val_mae isn't a
column). Retrains each of those best configs for a fixed number of epochs
(default 30) with early stopping effectively disabled (patience=epochs), and
writes results in the same json/xlsx pattern as the rest of the project to:

    search_results/<model>_<dataset>_finalcomparison.json
    search_results/<model>_<dataset>_finalcomparison.xlsx

Requires trainer.train() to accept a `patience` argument (added earlier).

Usage (from src/, as a script):
    python final_comparison.py --epochs 30

Usage (from a notebook / Colab cell):
    from final_comparison import run_final_comparison
    run_final_comparison(
        results_dir='/content/drive/MyDrive/AutoSTGNN/search_results',
        base_root='/content/drive/MyDrive/AutoSTGNN/data',
        epochs=30,
    )
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from trainer import train
from utils import results_to_excel


# Matches: <model>_<dataset>_<method>_<...anything else, e.g. rbf_ei_s42>_<date>.xlsx
# model/dataset are assumed to never contain underscores themselves, matching
# this project's naming (graphwavenet, dcrnn, stgcn, agcrn / metrla, pemsbay,
# pems04, pems08).
FILENAME_RE = re.compile(
    r'^(?P<model>[a-zA-Z0-9]+)_(?P<dataset>[a-zA-Z0-9]+)_(?P<method>.+)_(?P<date>\d{8})\.xlsx$'
)


def to_native(value):
    if isinstance(value, dict):
        return {k: to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def discover_result_files(results_dir):
    """Finds every <model>_<dataset>_<method>_<date>.xlsx in results_dir,
    grouped by (model, dataset). Skips files that don't match the naming
    convention (e.g. a prior finalcomparison.xlsx) rather than erroring."""
    groups = defaultdict(list)

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.xlsx'):
            continue
        m = FILENAME_RE.match(fname)
        if not m:
            print(f"  Skipping (doesn't match naming convention): {fname}")
            continue
        model = m.group('model').lower()
        dataset = m.group('dataset').lower()
        method = m.group('method')
        groups[(model, dataset)].append({
            'path': os.path.join(results_dir, fname),
            'method': method,
            'filename': fname,
        })

    return groups


KW_PREFIX = 'kw_'


def extract_best_config(file_info):
    """Reads one results .xlsx, returns (best_row_dict, val_mae_used) for
    whichever trial had the lowest val_mae (or best_val_mae as a fallback)."""
    df = pd.read_excel(file_info['path'])

    if 'val_mae' in df.columns and df['val_mae'].notna().any():
        val_col = 'val_mae'
    elif 'best_val_mae' in df.columns and df['best_val_mae'].notna().any():
        val_col = 'best_val_mae'
    else:
        print(f"  WARNING: no val_mae/best_val_mae column with data in "
              f"{file_info['filename']}, skipping")
        return None, None

    valid = df[df[val_col].notna()]
    if valid.empty:
        print(f"  WARNING: {val_col} is all-NaN in {file_info['filename']}, skipping")
        return None, None

    best_idx = valid[val_col].idxmin()
    best_row = valid.loc[best_idx].to_dict()
    return best_row, float(best_row[val_col])


def row_to_config(row):
    """Reconstructs lr, batch_size, model_kwargs from a flattened results row
    (kw_* columns -> model_kwargs dict)."""
    lr = float(row['lr'])
    batch_size = int(row['batch_size'])

    model_kwargs = {}
    for col, val in row.items():
        if isinstance(col, str) and col.startswith(KW_PREFIX):
            key = col[len(KW_PREFIX):]
            if isinstance(val, (np.integer, np.floating, np.bool_)):
                val = to_native(val)
            elif isinstance(val, float) and val.is_integer():
                # openpyxl/pandas often round-trips ints as floats
                pass
            model_kwargs[key] = val

    return lr, batch_size, model_kwargs


def run_final_comparison(
    results_dir='./search_results',
    base_root='./data',
    window=12,
    horizon=12,
    epochs=30,
    seed=42,
):
    os.makedirs(results_dir, exist_ok=True)
    groups = discover_result_files(results_dir)

    if not groups:
        print("No files matching <model>_<dataset>_<method>_<date>.xlsx found.")
        return {}

    all_outputs = {}

    for (model, dataset), files in groups.items():
        print(f"\n{'='*70}")
        print(f"=== {model} / {dataset} — {len(files)} method file(s) found ===")
        print('='*70)

        # If multiple files exist for the same method (re-runs / resumes on
        # different dates), keep only the most recent by filename's date.
        by_method = {}
        for f in files:
            existing = by_method.get(f['method'])
            if existing is None or f['filename'] > existing['filename']:
                by_method[f['method']] = f

        results_log = []
        search_start = time.time()

        for i, (method, file_info) in enumerate(sorted(by_method.items())):
            print(f"\n--- {method} ({file_info['filename']}) ---")

            best_row, val_mae_used = extract_best_config(file_info)
            if best_row is None:
                continue

            lr, batch_size, model_kwargs = row_to_config(best_row)
            print(f"Best config (val_mae={val_mae_used:.4f}): "
                  f"lr={lr}, batch_size={batch_size}, model_kwargs={model_kwargs}")

            trial_start = time.time()
            error = None
            try:
                predictor, pl_trainer, test_results, best_model_path, best_val_mae = train(
                    dataset_name=dataset,
                    model_name=model,
                    window=window,
                    horizon=horizon,
                    batch_size=batch_size,
                    lr=lr,
                    max_epochs=epochs,
                    base_root=base_root,
                    model_kwargs=model_kwargs,
                    seed=seed,
                    patience=epochs,  
                )

                test_results_dict = test_results[0]
                test_metrics = {
                    k: v for k, v in test_results_dict.items() if k.startswith('test_')
                }

                trial_record = {
                    'trial': i,
                    'method': method,
                    'source_file': file_info['filename'],
                    'search_val_mae': val_mae_used,
                    'lr': lr,
                    'batch_size': batch_size,
                    'model_kwargs': model_kwargs,
                    'best_model_path': best_model_path,
                    'best_val_mae': best_val_mae,
                    **test_metrics,
                }

            except Exception as e:
                print(f"  FAILED: {e}")
                trial_record = {
                    'trial': i,
                    'method': method,
                    'source_file': file_info['filename'],
                    'search_val_mae': val_mae_used,
                    'lr': lr,
                    'batch_size': batch_size,
                    'model_kwargs': model_kwargs,
                    'error': str(e),
                }

            trial_end = time.time()
            trial_record['trial_duration_sec'] = trial_end - trial_start
            trial_record['elapsed_since_start_sec'] = trial_end - search_start
            trial_record['dataset'] = dataset
            trial_record['model'] = model

            results_log.append(trial_record)

            out_path = os.path.join(results_dir, f"{model}_{dataset}_finalcomparison.json")
            with open(out_path, 'w') as f:
                json.dump(to_native(results_log), f, indent=2)
            try:
                results_to_excel(out_path)
            except Exception as e:
                print(f"  Excel export failed: {e}")

        all_outputs[(model, dataset)] = results_log

        valid = [r for r in results_log if 'test_mae' in r]
        if valid:
            best = min(valid, key=lambda r: r['test_mae'])
            print(f"\n{model}/{dataset} — best method: {best['method']} "
                  f"(test_mae={best['test_mae']:.4f})")

    return all_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Retrain each method's best config (by val_mae) to a fixed "
                     "epoch budget and produce a final side-by-side comparison."
    )
    parser.add_argument("--results_dir", type=str, default="./search_results")
    parser.add_argument("--base_root", type=str, default="./data")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_final_comparison(
        results_dir=args.results_dir,
        base_root=args.base_root,
        window=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()