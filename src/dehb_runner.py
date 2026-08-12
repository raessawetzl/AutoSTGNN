import os
import time
import json
import numpy as np
from datetime import datetime
from dehb import DEHB

from search_space import get_search_space
from trainer import train
from utils import results_to_excel

# This script uses DEHB (Differential Evolution Hyperband) to automatically
# search for the best hyperparameter configuration for a given STGNN model
# within a fixed time budget. Unlike random search which picks configurations
# blindly, DEHB uses multi-fidelity optimisation to focus compute on promising
# configurations and discard poor ones early, making it more efficient than
# random search.

MIN_FIDELITY = 3     # minimum epochs per trial. DEHB starts here for cheap evaluations
MAX_FIDELITY = 15    # maximum epochs per trial, only promising configurations reach this
RUNTIME_SECONDS = 7200  # total time budget for the DEHB search per model/dataset combination


def to_native(value):
    """Converts numpy types to native Python types for JSON serialisation."""
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


def run_dehb(
    model_name='dcrnn',
    dataset_name='metrla',
    window=12,
    horizon=12,
    base_root='./data',
    results_dir='./search_results',
    min_fidelity=MIN_FIDELITY,
    max_fidelity=MAX_FIDELITY,
    runtime_seconds=RUNTIME_SECONDS,
):
    """
    Pipeline manager for the DEHB search on one model and one dataset.
    Sets up the search space, initialises DEHB with the target function,
    and runs the search within the time budget. Tracks the best configuration
    found, saves a JSON log of all trials, and exports results to Excel.
    """
    os.makedirs(results_dir, exist_ok=True)
    cs = get_search_space(model_name)

    results_log = []
    search_start = time.time()
    trial_count = [0]

    print("=" * 50)
    print("DEHB optimisation: " + model_name + " on " + dataset_name)
    print("Min fidelity: " + str(min_fidelity) + " epochs")
    print("Max fidelity: " + str(max_fidelity) + " epochs")
    print("Runtime budget: " + str(runtime_seconds) + " seconds")
    print("=" * 50)

    def target_function(config, fidelity, **kwargs):
        """
        The function DEHB calls for every configuration it wants to evaluate.
        DEHB passes in the configuration and fidelity, and this function calls
        the shared trainer to do the actual training. It returns the validation
        MAE as the fitness value and the training time as the cost.
        """
        trial_count[0] += 1
        config_dict = {k: to_native(v) for k, v in dict(config).items()}

        lr = config_dict.pop('lr')
        batch_size = config_dict.pop('batch_size')
        model_kwargs = config_dict

        print("Trial " + str(trial_count[0]) +
              " | Fidelity: " + str(int(fidelity)) + " epochs" +
              " | lr=" + str(round(lr, 5)) +
              " | batch_size=" + str(batch_size))

        trial_start = time.time()

        try:
            predictor, trainer, test_results = train(
                dataset_name=dataset_name,
                model_name=model_name,
                window=window,
                horizon=horizon,
                batch_size=batch_size,
                lr=lr,
                max_epochs=int(fidelity),
                base_root=base_root,
                model_kwargs=model_kwargs,
            )

            val_mae = float(trainer.callback_metrics.get('val_mae', float('inf')))
            test_mae = test_results[0].get('test_mae', None)
            mae_15 = test_results[0].get('test_mae_at_15', None)
            mae_30 = test_results[0].get('test_mae_at_30', None)
            mae_60 = test_results[0].get('test_mae_at_60', None)

            print("  Val MAE: " + str(round(val_mae, 4)))

            trial_record = {
                'trial': trial_count[0],
                'fidelity': int(fidelity),
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'mae_at_15': mae_15,
                'mae_at_30': mae_30,
                'mae_at_60': mae_60,
            }

        except Exception as e:
            print("  Trial " + str(trial_count[0]) + " failed: " + str(e))
            val_mae = float('inf')
            trial_record = {
                'trial': trial_count[0],
                'fidelity': int(fidelity),
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'error': str(e),
            }

        trial_end = time.time()
        trial_record['trial_duration_sec'] = trial_end - trial_start
        trial_record['elapsed_since_start_sec'] = trial_end - search_start
        results_log.append(trial_record)

        timestamp = datetime.now().strftime('%Y%m%d')
        out_path = os.path.join(
            results_dir,
            model_name + "_" + dataset_name + "_dehb_" + timestamp + ".json"
        )
        with open(out_path, 'w') as f:
            json.dump(to_native(results_log), f, indent=2)

        try:
            results_to_excel(out_path)
        except Exception as e:
            print("Excel export failed: " + str(e))

        cost = trial_end - trial_start
        return {"fitness": val_mae, "cost": cost}

    dimensions = len(list(cs.values()))

    optimizer = DEHB(
        f=target_function,
        cs=cs,
        dimensions=dimensions,
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        eta=3,
        n_workers=1,
        output_path=results_dir,
    )

    optimizer.run(total_cost=runtime_seconds)

    valid_results = [r for r in results_log if 'val_mae' in r and 'error' not in r]
    if valid_results:
        best = min(valid_results, key=lambda r: r['val_mae'])
        print("Best trial: " + str(best['trial']) +
              " | Val MAE: " + str(round(best['val_mae'], 4)))
        print("Config: lr=" + str(best['lr']) +
              ", batch_size=" + str(best['batch_size']) +
              ", model_kwargs=" + str(best['model_kwargs']))
    else:
        print("No trials completed successfully.")

    return results_log


def main():
    datasets = ['metrla', 'pemsbay']
    for dataset_name in datasets:
        run_dehb(
            model_name='dcrnn',
            dataset_name=dataset_name,
        )


if __name__ == "__main__":
    main()