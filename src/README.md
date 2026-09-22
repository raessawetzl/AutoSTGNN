# Evaluating DEHB for Hyperparameter Optimisation of Spatio-Temporal Graph Neural Networks

**Charmaine Munyongani (MNYCHA010)**
CSC4002W Honours Research Project, University of Cape Town, 2026
**Supervisor:** Deshen Moodley

## Overview

This repository contains the code used for the individual component of the project. It includes the DEHB searches, Random Search baseline, reference configurations, and convergence experiments used to determine suitable fidelity ranges.

The project evaluates Differential Evolution Hyperband (DEHB) for hyperparameter optimisation of Spatio-Temporal Graph Neural Networks (STGNNs). DEHB is compared with Random Search and untuned reference configurations across three architectures:

* AGCRN
* STCN
* Graph WaveNet

The experiments use three datasets:

* METR-LA
* PEMS-BAY
* Electricity

DEHB and Random Search are compared using matched allocated epoch budgets.

## Files

| File                 | Purpose                                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataloader.py`      | Loads the datasets, constructs the graphs, performs the data splits and scaling, and creates the dataloaders used by the experiments.               |
| `trainer.py`         | Contains the shared training pipeline, model defaults, evaluation metrics, checkpointing, early stopping, and runtime tracking.                     |
| `search_space.py`    | Defines the hyperparameter search space for each architecture.                                                                                      |
| `utils.py`           | Contains helper functions for saving results, converting NumPy values, exporting results to Excel, and retrieving validation MAE.                   |
| `dehb_runner.py`     | Runs the DEHB search and retrains the selected configuration. Its `retrain_incumbent` function is also used for the final Random Search retraining. |
| `random_search.py`   | Runs Random Search at a fixed fidelity and records the results of each sampled configuration.                                                       |
| `retrain_rs.py`      | Selects the Random Search configuration with the lowest validation MAE and retrains it using the same final training function as DEHB.              |
| `train_baselines.py` | Shared group baseline script, included for completeness. The reference configurations reported here were run directly through `trainer.train`.        |
| `convergence.py`     | Trains sampled configurations across several epochs and records their MAE to help determine suitable fidelity ranges.                               |

`random_search_dcrnn.py` and `trainer_group_backup.py` are older files from earlier group work and are not used in the reported experiments.

## Environment

The experiments were run in Google Colab using A100 and T4 GPUs.

The main packages used were:

```text
Python 3.13
torch 2.11.0 (CUDA 12.8)
pytorch-lightning 2.6.5
torch-spatiotemporal (tsl) 0.9.5
ConfigSpace 1.2.2
dehb
numpy 1.26.4
```

The main dependencies can be installed with:

```bash
pip install torch-geometric
pip install torch-spatiotemporal dehb
pip install "pytorch-lightning==2.6.5"
pip install "numpy==1.26.4"
```

NumPy was pinned to version 1.26.4 for compatibility with the version of `torch-spatiotemporal` used in the experiments. When running in Colab, the runtime should be restarted after changing the NumPy version and before importing the project modules.

## Training Pipeline

All experiments use `trainer.train`, which provides a shared training and evaluation pipeline. This keeps the data processing, masked MAE loss, evaluation metrics, checkpointing, and random seed consistent across the different search methods.

A seed of 42 is used throughout the training pipeline.

The selected model checkpoint is determined using validation MAE. Test metrics are not used to select the incumbent configuration.

## Running the Reference Configuration

The reference configurations reported here were run directly through the shared `trainer.train` pipeline using the default model hyperparameters. For example, an AGCRN reference run on METR-LA can be started with:

```python
from trainer import train

predictor, trainer_obj, test_results, best_ckpt, best_val_mae = train(
    dataset_name='metrla',
    model_name='agcrn',
    window=12,
    horizon=12,
    batch_size=64,
    lr=0.001,
    max_epochs=30,
    base_root='./data',
    model_kwargs=None,
    patience=30,
    seed=42,
)
```

Here, `model_kwargs=None` uses the reference settings defined in `DEFAULT_MODEL_KWARGS`. The group's shared `train_baselines.py` script is left unchanged and is included for completeness.

## Running DEHB

A DEHB search followed by final incumbent retraining can be run using:

```python
from dehb_runner import run_dehb

run_dehb(
    model_name='agcrn',
    dataset_name='metrla',
    min_fidelity=3,
    max_fidelity=11,
    total_epoch_budget=275,
    retrain_epochs=30,
    base_root='./data',
    results_dir='./search_results',
)
```

The exact fidelity range and epoch budget depend on the architecture–dataset pair.

## Running Random Search

Random Search is first run at a fixed fidelity:

```python
from random_search import run_random_search

run_random_search(
    model_name='agcrn',
    dataset_name='metrla',
    n_trials=25,
    max_epochs=11,
    base_root='./data',
    results_dir='./search_results'
)
```

The configuration with the lowest validation MAE is then retrained using the same final retraining function used by DEHB:

```python
from retrain_rs import retrain_random_search

retrain_random_search(
    results_path='./search_results/agcrn_metrla_RS_<date>.json',
    model_name='agcrn',
    dataset_name='metrla',
    max_trials=25,
    fidelity=11,
    base_root='./data',
    results_dir='./search_results',
)
```

The `fidelity` argument records the fixed number of epochs allocated to each Random Search trial.

## Convergence Analysis

The convergence experiments were used to examine how validation MAE changed across training epochs and to help determine suitable fidelity ranges.

For example:

```python
from convergence import run_convergence_analysis

run_convergence_analysis(
    model_name='agcrn',
    dataset_name='metrla',
    n_configs=5,
    max_epochs=15,
    output_root='./convergence_plots/metrla'
)
```

Each dataset should use its own output directory to prevent plots from different datasets from overwriting one another.

## Experimental Settings

Random Search is given the same allocated epoch budget as DEHB for each
architecture–dataset pair. Random Search uses the maximum DEHB fidelity as
its fixed training fidelity.

| Dataset | Architecture | Min Fidelity | Max Fidelity | RS Trials | Allocated Epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| METR-LA | AGCRN, STCN | 3 | 11 | 25 | 275 |
| METR-LA | Graph WaveNet | 4 | 12 | 25 | 300 |
| PEMS-BAY | AGCRN, STCN | 3 | 11 | 20 | 220 |
| PEMS-BAY | Graph WaveNet | 4 | 12 | 20 | 240 |
| Electricity | AGCRN, STCN, Graph WaveNet | 4 | 12 | 20 | 240 |

DEHB uses `eta = 3`. Both methods select their incumbent using validation MAE.
The selected configuration is then retrained from scratch for 30 epochs using
the shared final training protocol.

If a Random Search log contains more trials than the allocated budget allows, `max_trials` is used to restrict the incumbent selection to the trials within the experimental budget. Trials are retained in order, so `max_trials=n` uses trials 0 to `n-1`.

## Implementation Notes

### Reproducibility

The shared training pipeline uses a seed of 42 for Python, NumPy, PyTorch, and PyTorch Lightning.

DEHB also receives its seed explicitly because its internal random number generation is separate from the global NumPy state.

### AGCRN and Graph Construction

AGCRN learns its spatial relationships through adaptive node embeddings and does not use the adjacency matrix constructed by `dataloader.py`.

The graph produced by the dataloader is therefore used by STCN and Graph WaveNet, while AGCRN learns its graph structure internally.

### Electricity Graph

Unlike METR-LA and PEMS-BAY, the Electricity dataset does not have a physical sensor network that can be used to construct an adjacency matrix.

Its graph is therefore constructed using the absolute Pearson correlation between the individual electricity series. For each node, the six most highly correlated series are retained as neighbours.

The Electricity series are also standardised individually because their scales can differ substantially.

### Random Search Fidelity

Random Search uses a fixed fidelity for every configuration, so the original Random Search logs do not store a separate fidelity value for each trial.

`retrain_rs.py` therefore accepts the fidelity as an argument and records the allocated search budget as:

```text
number of trials x fidelity
```

If no fidelity is supplied, the search budget is left empty rather than inferred.

### Allocated and Trained Epochs

The search budgets refer to **allocated epochs**. Individual training runs may finish earlier because of early stopping.

The DEHB and Random Search budget calculations both use allocated epochs so that the search effort is measured consistently.

### METR-LA AGCRN DEHB Run

The AGCRN DEHB search on METR-LA stopped after 216 of its allocated 275 epochs because it reached the wall-clock runtime safeguard after approximately 11.06 hours.

The runtime limit was increased for subsequent experiments. The reported search budget for this run therefore reflects the 216 epochs actually allocated before termination.

## Result Provenance

The DEHB results for all nine architecture–dataset combinations were produced using this implementation.

The Random Search retrains for AGCRN on PEMS-BAY and Electricity and STCN on PEMS-BAY were also produced using this code. The remaining Random Search retrains came from the shared group experiments and followed the same 30-epoch final training protocol and validation-based configuration selection.
