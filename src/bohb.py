import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier.hyperband import Hyperband
from ConfigSpace import Configuration

from search_space import get_search_space
from trainer import get_model, train  # reuse the shared team pipeline
from dataloader import get_dataloaders
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE

# ---------------------------------------------------------------
# settings
# ---------------------------------------------------------------
MODEL_NAME   = "stgcn"          # must match a key in trainer.MODEL_MAP
DATASET_NAME = "metrla"         # must match a key in dataloader.DATASET_MAP
N_TRIALS     = 50
MIN_BUDGET   = 5                # epochs
MAX_BUDGET   = 50               # epochs
SEED         = 42
OUTPUT_DIR   = Path(__file__).resolve().parent / "bohb_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# cached across trials so we don't reload data from disk every trial
_train_loader = None
_val_loader = None
_test_loader = None


def _ensure_data_loaded(batch_size: int):
    """(Re)loads dataloaders if batch_size changes between trials."""
    global _train_loader, _val_loader, _test_loader
    _train_loader, _val_loader, _test_loader = get_dataloaders(
        dataset_name=DATASET_NAME,
        batch_size=batch_size,
    )


def config_to_kwargs(config: dict) -> dict:
    """Splits a sampled config into (lr, batch_size, model_kwargs) — same
    convention used in the team's random_search.py."""
    config = dict(config)
    lr = float(config.pop("lr"))
    batch_size = int(config.pop("batch_size"))
    model_kwargs = {
        k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else v)
        for k, v in config.items()
    }
    return lr, batch_size, model_kwargs


def bohb_objective(config: Configuration, seed: int = SEED, budget: float = MAX_BUDGET) -> float:
    """
    Trains one config for `budget` epochs (short for low-budget rungs,
    longer for surviving configs). Returns validation MAE — SMAC minimises this.
    A failed/diverged trial returns a large finite value instead of crashing
    the whole search.
    """
    pl.seed_everything(seed)  # reproducibility, applies to init + shuffling

    try:
        lr, batch_size, model_kwargs = config_to_kwargs(config)
        _ensure_data_loaded(batch_size)

        sample_batch = next(iter(_train_loader))
        n_nodes = sample_batch.input.x.shape[2]
        input_size = sample_batch.input.x.shape[-1]
        output_size = sample_batch.target.y.shape[-1]

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
            metrics={"mae": MaskedMAE(), "mape": MaskedMAPE()},
        )

        checkpoint_cb = ModelCheckpoint(monitor="val_mae", mode="min", save_top_k=1)

        lightning_trainer = pl.Trainer(
            max_epochs=int(budget),
            accelerator="auto",
            devices=1,
            gradient_clip_val=5.0,          # guards against exploding-loss trials
            callbacks=[
                EarlyStopping(monitor="val_mae", patience=5, mode="min"),
                checkpoint_cb,
            ],
            enable_progress_bar=False,
            logger=False,
        )

        lightning_trainer.fit(
            predictor,
            train_dataloaders=_train_loader,
            val_dataloaders=_val_loader,
        )

        val_mae = lightning_trainer.callback_metrics.get("val_mae")
        val_mae = float(val_mae) if val_mae is not None else float("nan")

        if not np.isfinite(val_mae):
            print(f"  Trial diverged (val_mae={val_mae}) — penalising, not crashing")
            return 1e6

        return val_mae

    except Exception as e:
        print(f"  Trial failed with error: {e} — penalising, not crashing")
        return 1e6


def run_bohb():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cs = get_search_space(MODEL_NAME)

    scenario = Scenario(
        configspace=cs,
        name=f"BOHB_{MODEL_NAME}_{DATASET_NAME}",
        output_directory=OUTPUT_DIR,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=SEED,
        min_budget=MIN_BUDGET,
        max_budget=MAX_BUDGET,
    )

    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=bohb_objective,
        intensifier=intensifier,
        overwrite=True,
    )

    print("\n" + "=" * 60)
    print(f"Starting BOHB — {N_TRIALS} trials on {MODEL_NAME} / {DATASET_NAME}")
    print(f"Budget per trial: {MIN_BUDGET}-{MAX_BUDGET} epochs")
    print("=" * 60 + "\n")

    incumbent = smac.optimize()

    print("\n" + "=" * 60)
    print("BOHB complete. Best config found:")
    for key, val in dict(incumbent).items():
        print(f"  {key}: {val}")
    print("=" * 60)

    # -------------------------------------------------------------
    # retrain the winning config at full budget and get real test metrics,
    # using the team's shared train() so results are directly comparable
    # to baseline / random search runs
    # -------------------------------------------------------------
    lr, batch_size, model_kwargs = config_to_kwargs(dict(incumbent))

    print(f"\nRetraining best config for {MAX_BUDGET} epochs (full budget)...")
    predictor, lightning_trainer, test_results = train(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        window=12,
        horizon=12,
        batch_size=batch_size,
        lr=lr,
        max_epochs=MAX_BUDGET,
        model_kwargs=model_kwargs,
    )

    print(f"\nTest results with best BOHB config on {DATASET_NAME}:")
    print(test_results)

    results_path = OUTPUT_DIR / f"bohb_{MODEL_NAME}_{DATASET_NAME}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "best_config": {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(incumbent).items()},
            "test_results": test_results,
            "n_trials": N_TRIALS,
            "min_budget": MIN_BUDGET,
            "max_budget": MAX_BUDGET,
            "seed": SEED,
        }, f, indent=2)

    print(f"Results saved to {results_path}")
    return incumbent, test_results


if __name__ == "__main__":
    run_bohb()