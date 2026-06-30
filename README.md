

**step 1 — Preprocess the datasets**

```bash
python src/preprocessing/preprocess.py
```

This produces `PEMS04_processed.npz` and `PEMS08_processed.npz` in the `data/` folder, containing normalised train/val/test splits.

**step 2 - train STGCN with a fixed config**

```bash
python src/train_stgcn.py
```

**step 3 — run random search baseline**

```bash
python src/random_search_stgcn.py
```

**step 4 — run BOHB search (Bayesian Optimization + HyperBand)**
```bash
python src/bohb_runner.py
```
Runs a SMAC3-based BOHB search over the hyperparameter space defined below, using HyperBand to allocate training epochs adaptively across configurations. Trials are logged to Weights & Biases (offline mode by default — run `wandb sync` afterwards to upload). The best configuration found is retrained on the full budget and evaluated on the test set, with results saved to `src/bohb_results/bohb_STGCN_<DATASET_NAME>_results.txt`.

Key settings (in `bohb_runner.py`):
| Setting | Description |
|---|---|
| `DATASET_NAME` | `"PEMS04"` or `"PEMS08"` |
| `N_TRIALS` | Total number of configurations evaluated |
| `MIN_BUDGET` / `MAX_BUDGET` | Min/max epochs HyperBand allocates per trial |

> Note: this is compute-intensive — submit via `sbatch`.
---

## Hyperparameter Search Space

Defined in `src/search_space.py` using the ConfigSpace library. Shared across all three STGNN architectures:

| Hyperparameter | Type | Range |
|---|---|---|
| learning_rate | Float (log) | 1e-4 to 1e-2 |
| hidden_units | Integer | 32 to 256 |
| num_layers | Integer | 1 to 4 |
| dropout | Float | 0.0 to 0.5 |
| batch_size | Categorical | 16, 32, 64 |
| weight_decay | Float (log) | 1e-5 to 1e-3 |
| K_cheb (STGCN only) | Integer | 1 to 5 |

---

## Datasets

| Dataset | Region | Sensors | Period | Interval |
|---|---|---|---|---|
| PeMSD4 | San Francisco Bay Area | 307 | 2 months | 5 min |
| PeMSD8 | San Bernardino County, CA | 170 | 2 months | 5 min |

Both datasets are publicly available via the [LibCity](https://github.com/LibCity/Bigscity-LibCity) framework.

---


