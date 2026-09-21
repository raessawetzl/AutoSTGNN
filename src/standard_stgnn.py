"""
standard.py

trains a model with fixed, default hyperparameters (no search) across
multiple datasets, as a baseline to compare search results against.

for each dataset in DATASETS, trains MODEL_NAME for EPOCHS with LR/BATCH_SIZE
and no model_kwargs overrides, then logs the result (json + excel) after
every dataset so progress isn't lost on a crash.

usage:
    python standard.py

config:
    model_name  - model to train (agcrn, stgcn, graphwavenet, dcrnn)
    datasets    - list of datasets to train on (metrla, pemsbay, pems04, pems08, electricity)
    epochs      - training epochs per dataset
    lr          - learning rate
    batch_size  - batch size
    seed        - random seed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
from pathlib import Path
from datetime import datetime

from trainer import train
from utils import results_to_excel

MODEL_NAME  = "agcrn"           # options: agcrn, stgcn, graphwavenet, dcrnn
DATASETS    = ["metrla", "pemsbay", "electricity"]  # options: metrla, pemsbay, pems04, pems08, electricity
EPOCHS      = 30
SEED        = 42
LR          = 1e-3
BATCH_SIZE  = 64

BASE_DIR    = Path('/content/drive/MyDrive/AutoSTGNN')
DATA_ROOT   = str(BASE_DIR / 'data')
RESULTS_DIR = BASE_DIR / 'search_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_standard():
    """train MODEL_NAME with fixed hyperparameters on each dataset in DATASETS."""
    results_log = []
    search_start = time.time()

    for i, dataset in enumerate(DATASETS):
        print(f"\n=== Standard {MODEL_NAME} on {dataset} ===")
        trial_start = time.time()

        try:
            predictor, lightning_trainer, test_results, best_model_path, best_val_mae = train(
                dataset_name=dataset,
                model_name=MODEL_NAME,
                window=12,
                horizon=12,
                batch_size=BATCH_SIZE,
                lr=LR,
                max_epochs=EPOCHS,
                base_root=DATA_ROOT,
                model_kwargs=None,
                seed=SEED,
            )

            test_results_dict = test_results[0]
            test_metrics = {
                k: v for k, v in test_results_dict.items() if k.startswith('test_')
            }

            trial_record = {
                'trial': i,
                'algorithm': 'standard',
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'model_kwargs': {},
                'best_model_path': best_model_path,
                'best_val_mae': best_val_mae,
                **test_metrics,
            }

        except Exception as e:
            print(f"Trial {i} failed: {e}")
            trial_record = {
                'trial': i,
                'algorithm': 'standard',
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'model_kwargs': {},
                'error': str(e),
            }

        trial_end = time.time()
        trial_record['trial_duration_sec'] = trial_end - trial_start
        trial_record['elapsed_since_start_sec'] = trial_end - search_start
        trial_record['dataset'] = dataset

        results_log.append(trial_record)

        # write out after every dataset so a crash doesn't lose earlier results
        timestamp = datetime.now().strftime('%Y%m%d')
        out_path = RESULTS_DIR / f"{MODEL_NAME}_standard_{timestamp}.json"
        with open(out_path, 'w') as f:
            json.dump(results_log, f, indent=2, default=str)
        try:
            results_to_excel(str(out_path))
        except Exception as e:
            print(f"Excel export failed: {e}")

    return results_log


if __name__ == "__main__":
    run_standard()