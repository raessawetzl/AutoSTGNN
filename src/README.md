# AutoSTGNN

AutoSTGNN is a hyperparameter search and evaluation pipeline for spatiotemporal graph neural networks. It supports Graph WaveNet, DCRNN, STGCN, and AGCRN, and can be used with traffic and energy datasets such as METR-LA, PEMS-BAY, PEMS04, PEMS08, and Electricity.

The project is built around [torch-spatiotemporal (tsl)](https://github.com/TorchSpatiotemporal/tsl).

There are two hyperparameter search methods available:

- Random search
- BOHB (Bayesian Optimization + Hyperband) using SMAC

There are also scripts for retraining the best configurations, running fixed-hyperparameter baselines, and comparing the search methods.

## Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

`torch-spatiotemporal` uses `torch-geometric` and several compiled PyG packages such as scatter, sparse, and cluster. These need to match your PyTorch and CUDA versions.

If the installation fails, check the tsl installation instructions and install the appropriate PyTorch/PyG versions before trying again.

## Project files

| File | Description |
|---|---|
| `search_space.py` | Defines the hyperparameter search spaces for each model |
| `dataloader.py` | Loads the datasets and creates the graph connectivity and train/validation/test dataloaders |
| `trainer.py` | Creates the models and handles training with PyTorch Lightning |
| `utils.py` | Converts JSON result logs into Excel files |
| `random_search.py` / `main.py` | Runs the random hyperparameter search |
| `bohb.py` | Runs the BOHB search using SMAC and Hyperband |
| `rs_final.py` | Retrains the best random-search configuration for comparison with BOHB |
| `standard_stgnn.py` | Runs the models with fixed hyperparameters |
| `convergence.py` | Trains sampled configurations and plots MAE convergence |
| `plot.py` | Compares BOHB and random search results |

## Running the searches

### Random search

Run:

```bash
python main.py
```

The model, dataset, number of trials, and maximum number of epochs can be changed in `main.py`.

You can also call `run_random_search(...)` directly if you want to use it from another script.

### BOHB

Run:

```bash
python bohb.py
```

The main settings are defined near the top of the file, including:

- `MODEL_NAME`
- `DATASET_NAME`
- the budget settings
- checkpoint directories

The script was originally set up to run on Colab with Google Drive mounted at:

```
/content/drive/MyDrive/AutoSTGNN
```

If you're running it locally, change `BASE_DIR` and `CKPT_DIR` to the appropriate directories.

### Retraining the random-search result

To retrain the best configuration found by random search:

```bash
python rs_final.py
```

### Fixed-hyperparameter baseline

Run:

```bash
python standard_stgnn.py
```

This trains the models using the predefined hyperparameters rather than performing a search.

### Convergence analysis

`convergence.py` can be used to see how the MAE changes as different configurations are trained.

For example:

```bash
python convergence.py --model stgcn --dataset electricity --n_configs 5
```

### Comparing BOHB and random search

After running the searches, the results can be compared using:

```bash
python plot.py --results-dir ./results --outdir ./figures
```

This produces plots comparing the two search methods based on things such as the amount of compute/fidelity used and the best performance found over time.

## Notes

- `bohb.py`, `rs_final.py`, and `standard_stgnn.py` currently use the Google Drive path `/content/drive/MyDrive/AutoSTGNN`. Change `BASE_DIR` if you're running the project somewhere else.
- Search results are written to JSON after each trial and also converted to `.xlsx` files using `utils.results_to_excel`. This means results from completed trials are still available if a search stops unexpectedly.
- `plot.py` determines the search type, model, and dataset from the result filenames. It also handles some common naming variations using `ARCH_ALIASES` and `DATASET_ALIASES`. Keeping filenames reasonably close to the actual model and dataset names will make this work more reliably.
