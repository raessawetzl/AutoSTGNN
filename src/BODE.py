import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import argparse
import numpy as np
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from trainer import train  # tsl-based train() from your trainer.py


# ---------------------------------------------------------------------------
# ConfigSpace — one space per model, matching each tsl model's real signature
# ---------------------------------------------------------------------------
def get_config_space(model_name):
    model_name = model_name.lower()
    cs = CS.ConfigurationSpace(seed=42)

    # lr is shared and always tuned
    cs.add_hyperparameters([
        CSH.UniformFloatHyperparameter("lr", lower=1e-4, upper=1e-2, log=True),
    ])

    if model_name == "graphwavenet":
        cs.add_hyperparameters([
            CSH.CategoricalHyperparameter("hidden_size", [16, 32, 64]),
            CSH.CategoricalHyperparameter("ff_size", [128, 256, 512]),
            CSH.UniformIntegerHyperparameter("n_layers", lower=4, upper=10),
            CSH.UniformIntegerHyperparameter("emb_size", lower=5, upper=20),
            CSH.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5),
        ])

    elif model_name == "dcrnn":
        cs.add_hyperparameters([
            CSH.CategoricalHyperparameter("hidden_size", [16, 32, 64]),
            CSH.UniformIntegerHyperparameter("kernel_size", lower=1, upper=3),
            CSH.UniformIntegerHyperparameter("n_layers", lower=1, upper=3),
            CSH.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5),
        ])

    elif model_name == "stgcn":
        cs.add_hyperparameters([
            CSH.CategoricalHyperparameter("hidden_size", [32, 64, 128]),
            CSH.CategoricalHyperparameter("ff_size", [64, 128, 256]),
            CSH.UniformIntegerHyperparameter("n_layers", lower=1, upper=4),
            CSH.UniformIntegerHyperparameter("temporal_kernel_size", lower=2, upper=5),
            CSH.UniformIntegerHyperparameter("spatial_kernel_size", lower=1, upper=3),
            CSH.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5),
        ])

    elif model_name == "agcrn":
        # NOTE: dropout not included here — not confirmed as an AGCRNModel kwarg.
        # Verify with inspect.signature(AGCRNModel.__init__) before relying on this.
        cs.add_hyperparameters([
            CSH.CategoricalHyperparameter("hidden_size", [16, 32, 64]),
            CSH.UniformIntegerHyperparameter("emb_size", lower=5, upper=20),
            CSH.UniformIntegerHyperparameter("n_layers", lower=1, upper=3),
        ])

    else:
        raise ValueError(f"Unknown model '{model_name}'")

    return cs


# ---------------------------------------------------------------------------
# Config <-> vector conversion — now handles Categorical as well as Uniform types
# ---------------------------------------------------------------------------
def cs_to_bounds(cs):
    """Returns ordered list of hyperparameter descriptors for DE to operate on."""
    bounds = []
    for name, hp in cs.get_hyperparameters_dict().items():
        if isinstance(hp, CSH.UniformFloatHyperparameter):
            bounds.append({'name': name, 'type': 'float',
                            'lower': hp.lower, 'upper': hp.upper, 'log': hp.log})
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            bounds.append({'name': name, 'type': 'int',
                            'lower': float(hp.lower), 'upper': float(hp.upper), 'log': hp.log})
        elif isinstance(hp, CSH.CategoricalHyperparameter):
            bounds.append({'name': name, 'type': 'cat', 'choices': list(hp.choices)})
    return bounds


def vector_to_config(vector, bounds):
    """Convert a real-valued vector in [0,1]^D to a config dict."""
    config = {}
    for i, b in enumerate(bounds):
        v = float(np.clip(vector[i], 0.0, 1.0))
        if b['type'] == 'cat':
            n = len(b['choices'])
            idx = min(int(v * n), n - 1)  # map [0,1) -> index
            config[b['name']] = b['choices'][idx]
        else:
            lower, upper = b['lower'], b['upper']
            if b.get('log'):
                val = np.exp(np.log(lower) + v * (np.log(upper) - np.log(lower)))
            else:
                val = lower + v * (upper - lower)
            if b['type'] == 'int':
                val = int(round(val))
            config[b['name']] = val
    return config


def config_to_vector(config, bounds):
    """Convert a config dict to a normalised vector in [0,1]^D."""
    vector = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        val = config[b['name']]
        if b['type'] == 'cat':
            idx = b['choices'].index(val)
            vector[i] = (idx + 0.5) / len(b['choices'])  # center of that choice's bin
        else:
            lower, upper = b['lower'], b['upper']
            v = float(val)
            if b.get('log'):
                vector[i] = (np.log(v) - np.log(lower)) / (np.log(upper) - np.log(lower))
            else:
                vector[i] = (v - lower) / (upper - lower)
        vector[i] = float(np.clip(vector[i], 0.0, 1.0))
    return vector


def sample_random_vector(bounds):
    return np.random.uniform(0.0, 1.0, len(bounds))


# ---------------------------------------------------------------------------
# Expected Improvement acquisition function (unchanged)
# ---------------------------------------------------------------------------
def expected_improvement(X_new, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Differential Evolution inner loop to maximise EI (unchanged — Algorithm 2, lines 4-10)
# ---------------------------------------------------------------------------
def de_maximise_ei(gp, y_best, bounds, n_pop=10, k=20, f=0.8, p_c=0.9):
    D = len(bounds)
    population = np.random.uniform(0.0, 1.0, (n_pop, D))
    ei_vals = expected_improvement(population, gp, y_best)

    for iteration in range(k):
        new_population = population.copy()
        new_ei_vals = ei_vals.copy()

        for i in range(n_pop):
            candidates = [j for j in range(n_pop) if j != i]
            x1, x2, x3 = np.random.choice(candidates, 3, replace=False)

            donor = population[x1] + f * (population[x2] - population[x3])
            donor = np.clip(donor, 0.0, 1.0)

            delta = np.random.randint(0, D)
            r = np.random.uniform(0.0, 1.0, D)
            trial = np.where((r <= p_c) | (np.arange(D) == delta), donor, population[i])

            ei_trial = expected_improvement(trial.reshape(1, -1), gp, y_best)[0]
            ei_target = expected_improvement(population[i].reshape(1, -1), gp, y_best)[0]

            if ei_trial >= ei_target:
                new_population[i] = trial
                new_ei_vals[i] = ei_trial
            else:
                new_population[i] = population[i]
                new_ei_vals[i] = ei_target

        population = new_population
        ei_vals = new_ei_vals

    best_idx = np.argmax(ei_vals)
    return population[best_idx]


# ---------------------------------------------------------------------------
# Single training run — now delegates to trainer.train() (tsl Predictor/Trainer)
# ---------------------------------------------------------------------------
def train_one_run(base_args, config, dataset_name):
    config = dict(config)  # avoid mutating caller's dict
    lr = float(config.pop('lr'))
    model_kwargs = config  # everything remaining is model-specific

    predictor, pl_trainer, test_results = train(
        dataset_name=dataset_name,
        model_name=base_args.model,
        window=base_args.window,
        horizon=base_args.horizon,
        batch_size=base_args.batch_size,
        lr=lr,
        max_epochs=base_args.epochs,
        base_root=base_args.base_root,
        model_kwargs=model_kwargs,
    )

    # Lightning logs 'val_mae' automatically from Predictor's metrics dict during validation.
    # Falls back to test_mae if val_mae wasn't logged for some reason.
    val_mae = pl_trainer.callback_metrics.get('val_mae')
    if val_mae is not None:
        val_mae = float(val_mae)
    else:
        val_mae = float(test_results[0].get('test_mae', float('inf')))

    return val_mae


# ---------------------------------------------------------------------------
# BO-DE main loop (Algorithm 2) — structure unchanged
# ---------------------------------------------------------------------------
def bo_de(base_args, dataset_name, T, n_init, n_pop, k, f, p_c):
    cs = get_config_space(base_args.model)
    bounds = cs_to_bounds(cs)

    X_obs = []
    y_obs = []
    results = []

    best_mae = float("inf")
    best_config = None

    print(f"=== Initialising with {n_init} random observations ===", flush=True)
    for i in range(n_init):
        vec = sample_random_vector(bounds)
        config = vector_to_config(vec, bounds)
        print(f"\n--- Init {i + 1}/{n_init} ---", flush=True)
        print("Config:", config, flush=True)

        try:
            val_mae = train_one_run(base_args, config, dataset_name)
        except Exception as e:
            print(f"Init {i + 1} failed: {e}", flush=True)
            val_mae = float("inf")

        X_obs.append(vec)
        y_obs.append(val_mae)
        results.append({"phase": "init", "config": config, "val_mae": val_mae})
        print(f"Init {i + 1} Val MAE: {val_mae:.4f}", flush=True)

        if val_mae < best_mae:
            best_mae = val_mae
            best_config = config
            print(f"*** New best: {best_mae:.4f} ***", flush=True)

    print(f"\n=== Starting BO-DE for {T} iterations ===", flush=True)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
        n_restarts_optimizer=5,
        normalize_y=True,
    )

    for t in range(1, T + 1):
        print(f"\n--- BO-DE Iteration {t}/{T} ---", flush=True)

        X_arr = np.array(X_obs)
        y_arr = np.array(y_obs)

        y_arr_fit = np.where(
            np.isinf(y_arr),
            np.nanmax(y_arr[~np.isinf(y_arr)]) * 2 if np.any(~np.isinf(y_arr)) else 1e6,
            y_arr
        )

        gp.fit(X_arr, y_arr_fit)
        y_best = float(np.min(y_arr_fit))

        best_vec = de_maximise_ei(gp, y_best, bounds, n_pop=n_pop, k=k, f=f, p_c=p_c)
        config = vector_to_config(best_vec, bounds)
        print("Next config (from DE):", config, flush=True)

        try:
            val_mae = train_one_run(base_args, config, dataset_name)
        except Exception as e:
            print(f"Iteration {t} failed: {e}", flush=True)
            val_mae = float("inf")

        X_obs.append(best_vec)
        y_obs.append(val_mae)
        results.append({"phase": "bode", "iteration": t, "config": config, "val_mae": val_mae})
        print(f"Iteration {t} Val MAE: {val_mae:.4f}", flush=True)

        if val_mae < best_mae:
            best_mae = val_mae
            best_config = config
            print(f"*** New best: {best_mae:.4f} ***", flush=True)

    return best_config, best_mae, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["metrla", "pemsbay", "pems04", "pems08"])
    parser.add_argument("--model", type=str, required=True,
                         choices=["graphwavenet", "dcrnn", "stgcn", "agcrn"])

    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--n_pop", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--f", type=float, default=0.8)
    parser.add_argument("--p_c", type=float, default=0.9)

    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base_root", type=str, default="./data")

    args = parser.parse_args()

    print("Model:", args.model, flush=True)
    print("Dataset:", args.dataset, flush=True)
    print(f"BO-DE: T={args.T}, n_init={args.n_init}, n_pop={args.n_pop}, "
          f"k={args.k}, f={args.f}, p_c={args.p_c}", flush=True)

    best_config, best_mae, results = bo_de(
        args, args.dataset,
        T=args.T, n_init=args.n_init, n_pop=args.n_pop,
        k=args.k, f=args.f, p_c=args.p_c
    )

    print("\n========== BO-DE Complete ==========", flush=True)
    print(f"Best Val MAE: {best_mae:.4f}", flush=True)
    print("Best Config:", best_config, flush=True)

    results_path = f"{args.dataset}_{args.model}_bode_results.json"
    with open(results_path, "w") as fp:
        json.dump({
            "best_config": best_config,
            "best_mae": best_mae,
            "trials": results
        }, fp, indent=2)
    print("Results saved to:", results_path, flush=True)


if __name__ == "__main__":
    main()