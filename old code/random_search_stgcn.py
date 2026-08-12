import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

from search_space import get_stgcn_config_space
from train_stcn import train 


def sample_config(cs):
    """Sample one config from the ConfigSpace and cast types cleanly."""
    config = cs.sample_configuration()
    return {k: int(v) if isinstance(v, np.integer) else v for k, v in dict(config).items()}


def random_search(n_configs=20, epochs=15, dataset_name="METR-LA"):
    cs = get_stgcn_config_space()
    results = []

    for i in range(n_configs):
        config = sample_config(cs)
        print(f"\n{'='*60}")
        print(f"Random Search — Config {i+1}/{n_configs} — {dataset_name}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        # train() already: masks loss correctly, seeds for reproducibility,
        # reloads best checkpoint, evaluates per-horizon, and logs to
        # results/stcn_baseline_results.txt + results/results.csv
        best_val_mae = train(
            config,
            dataset_name=dataset_name,
            epochs=epochs,
            algorithm="random_search"
        )

        results.append((best_val_mae, config))
        print(f"Best val MAE for this config: {best_val_mae:.4f}")

    results.sort(key=lambda x: x[0])

    print(f"\n{'='*60}")
    print(f"Top 5 configs — {dataset_name}")
    print(f"{'='*60}")
    for val_mae, cfg in results[:5]:
        print(f"Val MAE: {val_mae:.4f} | {cfg}")

    return results


def save_summary(results, dataset_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rs_{dataset_name.lower().replace('-', '')}_summary.txt"
    with open(path, "w") as f:
        f.write(f"Random Search Results — {dataset_name}\n")
        f.write("=" * 50 + "\n")
        for val_mae, cfg in results:
            f.write(f"Val MAE: {val_mae:.4f} | {cfg}\n")
    print(f"Summary saved to {path}")


if __name__ == "__main__":
    N_CONFIGS = 20
    EPOCHS = 15
    RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

    print("-" * 60)
    print("Random Search — METR-LA")
    print("-" * 60)
    results_metrla = random_search(n_configs=N_CONFIGS, epochs=EPOCHS, dataset_name="METR-LA")
    save_summary(results_metrla, "METR-LA", RESULTS_DIR)

    print("\n" + "-" * 60)
    print("Random Search — PEMS-BAY")
    print("-" * 60)
    results_pemsbay = random_search(n_configs=N_CONFIGS, epochs=EPOCHS, dataset_name="PEMS-BAY")
    save_summary(results_pemsbay, "PEMS-BAY", RESULTS_DIR)