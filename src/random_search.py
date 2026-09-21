"""
random_search.py

random hyperparameter search for stgnn models.

samples n_trials random configs from a model's search space, trains and
tests each one, and logs results (json + excel) after every trial so
progress isn't lost on a crash. supports resuming a previous run.

usage:
    from random_search import run_random_search
    run_random_search(model_name='stgcn', dataset_name='electricity', n_trials=20, max_epochs=12)

config:
    model_name    - model to search over (dcrnn, stgcn, graphwavenet, agcrn)
    dataset_name  - dataset to train/eval on (metrla, pemsbay, pems04, pems08, electricity)
    n_trials      - number of random configs to try
    max_epochs    - max training epochs per trial
    window        - input window length
    horizon       - forecast horizon length
    base_root     - root directory containing the dataset
    results_dir   - directory to write per-trial results (json/excel) to
    resume_from   - path to a previous results json to resume from
    patience      - early-stopping patience (epochs)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import results_to_excel
from search_space import get_search_space
from trainer import train


def to_native(value):
    """recursively convert numpy scalars/arrays to native python types for json serialization."""
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


def run_random_search(
    model_name='dcrnn',       # options: dcrnn, stgcn, graphwavenet, agcrn
    dataset_name='metrla',    # options: metrla, pemsbay, pems04, pems08, electricity
    n_trials=20,
    max_epochs=20,
    window=12,
    horizon=12,
    base_root='./data',
    results_dir=Path(__file__).resolve().parent / 'search_results',
    resume_from=None,
    patience=30,
):
    """sample and train n_trials random configs, logging results after each trial."""
    os.makedirs(results_dir, exist_ok=True)

    cs = get_search_space(model_name)
    configs = cs.sample_configuration(n_trials)
    if n_trials == 1:
        configs = [configs]

    results_log = []
    start_trial = 0

    if resume_from and os.path.exists(resume_from):
        with open(resume_from, 'r') as f:
            results_log = json.load(f)
        start_trial = len(results_log)
        print(f"Resuming from trial {start_trial} (found {len(results_log)} completed trials)")

    search_start = time.time()

    for i, config in enumerate(configs):
        if i < start_trial:
            continue  # already completed in a previous (resumed) run

        config_dict = {k: to_native(v) for k, v in dict(config).items()}
        lr = config_dict.pop('lr')
        batch_size = config_dict.pop('batch_size')
        model_kwargs = config_dict

        print(f"\n=== Trial {i+1}/{n_trials} ===")
        print(f"lr={lr}, batch_size={batch_size}, model_kwargs={model_kwargs}")

        trial_start = time.time()

        try:
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
                patience=patience,
            )

            test_results_dict = test_results[0]
            test_metrics = {
                k: v for k, v in test_results_dict.items() if k.startswith('test_')
            }

            trial_record = {
                'trial': i,
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'best_model_path': best_model_path,
                'best_val_mae': best_val_mae,
                **test_metrics,
            }

        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            trial_record = {
                'trial': i,
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'error': str(e),
            }

        trial_end = time.time()
        trial_record['trial_duration_sec'] = trial_end - trial_start
        trial_record['elapsed_since_start_sec'] = trial_end - search_start

        results_log.append(trial_record)

        # write out after every trial so a crash doesn't lose earlier results
        timestamp = datetime.now().strftime('%Y%m%d')
        out_path = os.path.join(
            results_dir, f"{model_name}_{dataset_name}_RS_{timestamp}.json"
        )
        with open(out_path, 'w') as f:
            json.dump(to_native(results_log), f, indent=2)
        try:
            results_to_excel(out_path)
        except Exception as e:
            print(f"Excel export failed: {e}")

    valid_results = [r for r in results_log if 'test_mae' in r]
    if valid_results:
        best = min(valid_results, key=lambda r: r['test_mae'])
        print(f"\nBest trial: {best['trial']} with test_mae={best['test_mae']}")
        print(f"Config: lr={best['lr']}, batch_size={best['batch_size']}, model_kwargs={best['model_kwargs']}")
        print(f"Found at {best['elapsed_since_start_sec']:.1f}s into the search")
    else:
        print("\nNo trials completed successfully.")

    return results_log