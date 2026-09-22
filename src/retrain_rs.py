"""
Retrains the best Random Search configuration using the same final
training protocol as DEHB.
"""

import json
import argparse

from dehb_runner import get_incumbent, total_overhead, retrain_incumbent

RETRAIN_EPOCHS = 30
RETRAIN_PATIENCE = 30


def load_trials(results_path, max_trials=None, fidelity=None):
    """Loads and prepares the Random Search trials for incumbent selection."""
    with open(results_path, 'r') as f:
        trials = json.load(f)

    trials = [t for t in trials if isinstance(t.get('trial'), int)]

    # The validation MAE key is matched to the format used by the DEHB helpers
    for t in trials:
        if 'val_mae' not in t and 'best_val_mae' in t:
            t['val_mae'] = t['best_val_mae']
        t.setdefault('fidelity', fidelity)

    trials = sorted(trials, key=lambda t: t['trial'])

    if max_trials is not None:
        trials = trials[:max_trials]

    return trials


def retrain_random_search(
    results_path,
    model_name,
    dataset_name,
    base_root='./data',
    results_dir='./search_results',
    window=12,
    horizon=12,
    retrain_epochs=RETRAIN_EPOCHS,
    retrain_patience=RETRAIN_PATIENCE,
    max_trials=None,
    fidelity=None,
):
    """Selects the Random Search incumbent and retrains it using the final protocol."""
    trials = load_trials(
        results_path,
        max_trials=max_trials,
        fidelity=fidelity
    )

    best = get_incumbent(trials)

    if best is None:
        print("No trials completed successfully in " + results_path)
        return None

    # The search budget is the number of trials multiplied by the fixed fidelity
    epochs_used = len(trials) * fidelity if fidelity else None
    wallclock = sum(t.get('trial_duration_sec', 0) for t in trials)

    print(
        "Random search incumbent: trial " + str(best['trial']) +
        " | Val MAE: " + str(round(best['val_mae'], 4)) +
        " | over " + str(len(trials)) + " trials" +
        " | budget " + str(epochs_used) + " epochs"
    )

    return retrain_incumbent(
        best=best,
        model_name=model_name,
        dataset_name=dataset_name,
        window=window,
        horizon=horizon,
        base_root=base_root,
        results_dir=results_dir,
        retrain_epochs=retrain_epochs,
        search_epochs_used=epochs_used,
        search_trials=len(trials),
        search_wallclock_sec=wallclock,
        search_overhead_sec=total_overhead(trials),
        search_method='random_search',
        retrain_patience=retrain_patience,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the random search incumbent"
    )

    parser.add_argument("--results_path", type=str, required=True)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["graphwavenet", "dcrnn", "stgcn", "agcrn"]
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["metrla", "pemsbay", "electricity", "pems04", "pems08"]
    )

    parser.add_argument("--base_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./search_results")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--retrain_epochs", type=int, default=RETRAIN_EPOCHS)
    parser.add_argument("--retrain_patience", type=int, default=RETRAIN_PATIENCE)
    parser.add_argument("--max_trials", type=int, default=None)

    parser.add_argument(
        "--fidelity",
        type=int,
        default=None,
        help="Epochs used for each Random Search trial"
    )

    args = parser.parse_args()

    retrain_random_search(
        results_path=args.results_path,
        model_name=args.model,
        dataset_name=args.dataset,
        base_root=args.base_root,
        results_dir=args.results_dir,
        window=args.window,
        horizon=args.horizon,
        retrain_epochs=args.retrain_epochs,
        retrain_patience=args.retrain_patience,
        max_trials=args.max_trials,
        fidelity=args.fidelity,
    )


if __name__ == "__main__":
    main()