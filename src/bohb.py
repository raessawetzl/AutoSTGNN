import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl

import hashlib #checkpoint resumption

from smac import MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.initial_design.random_design import RandomInitialDesign
from smac.main.config_selector import ConfigSelector
from ConfigSpace import Configuration

from utils import results_to_excel
from search_space import get_search_space
from trainer import get_model, train, build_metrics
from dataloader import get_dataloaders
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE

from smac.callback import Callback

# settings ---------------------------------------------------------------
MODEL_NAME   = "graphwavenet"
DATASET_NAME = "metrla"

ETA          = 3
MIN_BUDGET   = 4        
MAX_BUDGET   = 12
N_TRIALS       = 10000  
MAX_TOTAL_EPOCHS = 240   
FINAL_EPOCHS = 30     
CRASH_COST   = 100.0
SEED         = 42

BASE_DIR     = Path('/content/drive/MyDrive/AutoSTGNN')
OUTPUT_DIR   = BASE_DIR / 'bohb_results'
DATA_ROOT    = str(BASE_DIR / 'data')

# ------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

_loader_cache = {}      # batch_size -> (train, val, test)
_shapes = {}            # batch_size -> (n_nodes, input_size, output_size)
_train_loader = None
_val_loader = None
_test_loader = None

_results_log = []
_trial_counter = 0
_search_start_time = None
_time_offset = 0.0

LOG_PATH = OUTPUT_DIR / f"bohb_{MODEL_NAME}_{DATASET_NAME}_seed{SEED}.json"

#checkpoint resumption
CKPT_DIR = Path("/content/rung_ckpts")   # local disk, NOT Drive — see note
_rung_state = {}                          # config_key -> {"epochs", "best", "snapshot"}

def _config_key(config) -> str:
    blob = json.dumps({k: v for k, v in sorted(dict(config).items())},
                      sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:16]

class BestScore(pl.Callback):
    """Tracks best val_mae and snapshots every val_ metric at that epoch."""
    def __init__(self, best=float("inf"), snapshot=None):
        self.best = best
        self.snapshot = snapshot or {}

    def on_validation_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("val_mae")
        if v is None:
            return
        v = float(v)
        if v < self.best:
            self.best = v
            self.snapshot = {
                k: float(x) for k, x in trainer.callback_metrics.items()
                if k.startswith("val_")
            }


def _ensure_data_loaded(batch_size: int):
    """Builds dataloaders once per batch size and reuses them thereafter."""
    global _train_loader, _val_loader, _test_loader

    if batch_size not in _loader_cache:
        _loader_cache[batch_size] = get_dataloaders(
            dataset_name=DATASET_NAME,
            batch_size=batch_size,
            base_root=DATA_ROOT,
        )
        sample_batch = next(iter(_loader_cache[batch_size][0]))
        _shapes[batch_size] = (
            sample_batch.input.x.shape[2],
            sample_batch.input.x.shape[-1],
            sample_batch.target.y.shape[-1],
        )

    _train_loader, _val_loader, _test_loader = _loader_cache[batch_size]


def config_to_kwargs(config: dict) -> dict:
    """Splits a sampled config into (lr, batch_size, model_kwargs)."""
    config = dict(config)
    lr = float(config.pop("lr"))
    batch_size = int(config.pop("batch_size"))
    model_kwargs = {
        k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else v)
        for k, v in config.items()
    }
    return lr, batch_size, model_kwargs


def bohb_objective(config: Configuration, seed: int = 0, budget: float | None = None) -> float:
    """Trains one config for `budget` epochs and returns its best validation MAE."""
    global _trial_counter
    _trial_counter += 1
    trial_num = _trial_counter
    trial_start = time.time()

    epochs = MAX_BUDGET if budget is None else max(1, int(round(budget)))

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    lr, batch_size, model_kwargs = config_to_kwargs(config)
    key = _config_key(config) 

    try:
        _ensure_data_loaded(batch_size)
        n_nodes, input_size, output_size = _shapes[batch_size]

        model, _ = get_model(
            model_name=MODEL_NAME,
            n_nodes=n_nodes,
            input_size=input_size,
            output_size=output_size,
            horizon=12,
            model_kwargs=model_kwargs,
        )

        predictor = Predictor(
            model=model,
            optim_class=torch.optim.Adam,
            optim_kwargs={"lr": lr},
            loss_fn=MaskedMAE(),
            metrics=build_metrics(),
        )

        ckpt = CKPT_DIR / f"{key}.ckpt"
        prev = _rung_state.get(key, {"epochs": 0, "best": float("inf"), "snapshot": {}})

        if prev["epochs"] >= epochs:
            # already trained at least this far — reuse, spend zero epochs
            new_epochs = 0
            val_mae = prev["best"]
            best_cb = BestScore(prev["best"], prev["snapshot"])
        else:
            resume_path = str(ckpt) if ckpt.exists() else None
            best_cb = BestScore(prev["best"], prev["snapshot"])

            lightning_trainer = pl.Trainer(
                max_epochs=epochs,          # absolute target; Lightning restores the counter
                accelerator="auto",
                devices=1,
                num_sanity_val_steps=0,
                callbacks=[best_cb],
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )

            lightning_trainer.fit(
                predictor,
                train_dataloaders=_train_loader,
                val_dataloaders=_val_loader,
                ckpt_path=resume_path,      # None on the first rung
            )

            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            lightning_trainer.save_checkpoint(ckpt)
            new_epochs = epochs - prev["epochs"]

        _rung_state[key] = {"epochs": epochs, "best": best_cb.best, "snapshot": best_cb.snapshot}
        val_mae = best_cb.best
        duration = time.time() - trial_start

        if not np.isfinite(val_mae):
            raise RuntimeError(f"diverged: best val_mae={val_mae}")

        print(f"  Trial {trial_num} (budget={epochs} epochs): val_mae={val_mae:.4f}")
        _log_trial({
            "trial": trial_num,
            "algorithm": "bohb",
            "budget_epochs": epochs,
            "epochs_run": new_epochs,              # was lightning_trainer.current_epoch
            "epochs_cumulative_for_config": epochs,
            "status": "ok",
            "seed": SEED,
            "lr": lr,
            "batch_size": batch_size,
            "model_kwargs": model_kwargs,
            "val_mae": val_mae,
            **best_cb.snapshot,
            "trial_duration_sec": duration,
        })
        return val_mae

    except Exception as e:
        duration = time.time() - trial_start
        print(f"  Trial {trial_num} failed: {e} — recording as crashed")
        _log_trial({
            "trial": trial_num,
            "algorithm": "bohb",
            "budget_epochs": epochs,
            "epochs_run": None,
            "status": "failed",
            "seed": SEED,
            "lr": lr,
            "batch_size": batch_size,
            "model_kwargs": model_kwargs,
            "val_mae": None,
            "error": str(e),
            "trial_duration_sec": duration,
        })
        raise


def _load_existing_log():
    global _results_log, _trial_counter, _time_offset

    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            _results_log = json.load(f)
        nums = [r["trial"] for r in _results_log if isinstance(r.get("trial"), int)]
        _trial_counter = max(nums) if nums else 0
        _time_offset = max(
            (r.get("elapsed_since_start_sec") or 0.0) for r in _results_log
        ) if _results_log else 0.0
        print(f"Resumed: {len(_results_log)} entries, counter {_trial_counter}, "
              f"offset {_time_offset/3600:.2f}h")


def _log_trial(trial_record):
    """Appends one trial to the results log and writes JSON + Excel."""
    global _search_start_time
    if _search_start_time is None:
        _search_start_time = time.time()

    trial_record["elapsed_since_start_sec"] = _time_offset + (time.time() - _search_start_time)
    trial_record["cumulative_epochs"] = (
        sum(r.get("epochs_run") or 0 for r in _results_log)
        + (trial_record.get("epochs_run") or 0)
    )
    trial_record["logged_at"] = datetime.now().isoformat(timespec="seconds")

    _results_log.append(trial_record)

    with open(LOG_PATH, "w") as f:
        json.dump(_results_log, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    try:
        results_to_excel(str(LOG_PATH))
    except Exception as e:
        print(f"Excel export failed: {e}")

class EpochBudgetStopper(Callback):
    """Stops SMAC once cumulative trained epochs hit max_total_epochs."""
    def __init__(self, max_total_epochs):
        self.max_total_epochs = max_total_epochs

    def on_tell_end(self, smbo, info, value):
        total = sum(r.get("epochs_run") or 0 for r in _results_log)
        if total >= self.max_total_epochs:
            print(f"\nReached epoch budget: {total}/{self.max_total_epochs} epochs. Stopping BOHB.")
            return False   # returning False halts smac.optimize()
        return True

def run_bohb():
    global _search_start_time
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _load_existing_log()
    _search_start_time = time.time()

    cs = get_search_space(MODEL_NAME)
    n_hps = len(list(cs.values()))

    scenario = Scenario(
        configspace=cs,
        name=f"BOHB_{MODEL_NAME}_{DATASET_NAME}_seed{SEED}",
        output_directory=OUTPUT_DIR,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=SEED,
        min_budget=MIN_BUDGET,
        max_budget=MAX_BUDGET,
        crash_cost=CRASH_COST,
    )

    intensifier = Hyperband(
        scenario,
        eta=ETA,
        incumbent_selection="highest_budget",
    )

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=bohb_objective,
        intensifier=intensifier,
        random_design=ProbabilityRandomDesign(probability=1/3, seed=SEED),
        initial_design=RandomInitialDesign(scenario, n_configs=n_hps + 2, seed=SEED),
        config_selector=ConfigSelector(scenario, retrain_after=1),
        overwrite=False,     # resumes from OUTPUT_DIR instead of wiping it
        callbacks=[EpochBudgetStopper(MAX_TOTAL_EPOCHS)],
    )

    print("\n" + "=" * 60)
    print(f"BOHB — {N_TRIALS} trials on {MODEL_NAME} / {DATASET_NAME}")
    print(f"Budget ladder (eta={ETA}): {MIN_BUDGET} -> {MAX_BUDGET} epochs")
    print(f"Seed: {SEED}   Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    print("Pre-flight: ensuring dataset is available...")
    _ensure_data_loaded(64)
    print(f"Dataset OK — {_shapes[64][0]} nodes\n")

    incumbent = smac.optimize()

    print("\n" + "=" * 60)
    print("BOHB complete. Best config found:")
    for key, val in dict(incumbent).items():
        print(f"  {key}: {val}")
    print("=" * 60)

    lr, batch_size, model_kwargs = config_to_kwargs(dict(incumbent))

    final_start = time.time()
    print(f"\nRetraining best config for {FINAL_EPOCHS} epochs...")
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
    final_duration = time.time() - final_start

    print(f"\nTest results with best BOHB config on {DATASET_NAME}:")
    print(test_results)

    test_results_dict = test_results[0]
    test_metrics = {k: v for k, v in test_results_dict.items() if k.startswith('test_')}

    final_record = {
        'trial': 'FINAL',
        'algorithm': 'bohb',
        'seed': SEED,
        'budget_epochs': FINAL_EPOCHS,
        'epochs_run': lightning_trainer.current_epoch,
        'status': 'ok',
        'trial_duration_sec': final_duration,
        'lr': lr,
        'batch_size': batch_size,
        'model_kwargs': model_kwargs,
        'best_model_path': best_model_path,
        'best_val_mae': best_val_mae,
        **test_metrics,
    }
    _log_trial(final_record)

    results_path = OUTPUT_DIR / f"bohb_{MODEL_NAME}_{DATASET_NAME}_seed{SEED}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "best_config": {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in dict(incumbent).items()},
            "test_results": test_results,
            "n_trials": N_TRIALS,
            "eta": ETA,
            "min_budget": MIN_BUDGET,
            "max_budget": MAX_BUDGET,
            "final_epochs": FINAL_EPOCHS,
            "seed": SEED,
        }, f, indent=2)

    print(f"Results saved to {results_path}")
    return incumbent, test_results


if __name__ == "__main__":
    run_bohb()