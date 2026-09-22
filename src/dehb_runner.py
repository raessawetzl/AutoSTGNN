import os
import time
from datetime import datetime
from dehb import DEHB

from search_space import get_search_space
from trainer import train, LAST_RUN_TIMINGS
from utils import to_native, get_best_val_mae, save_json, results_to_excel

# DEHB searches for the best hyperparameter configuration using validation MAE
# The search budget is measured in training epochs for consistency across methods 
# Trial timings are also recorded to compare computational costs 
# The best configuration is retrained from scratch for final evaluation

MIN_FIDELITY = 3          # minimum epochs per trial. DEHB starts here for cheap evaluations
MAX_FIDELITY = 20         # maximum epochs per trial
TOTAL_EPOCH_BUDGET = 300  # total training epochs across the whole search
RETRAIN_EPOCHS = 100      # Maximum number of epochs used to retrain the best configuration
RUNTIME_SECONDS = 86400   # Maximum runtime allowed for the search


class BudgetExhausted(Exception):
    """Indicates that the total epoch budget has been reached."""
    pass


def run_dehb(
    model_name='dcrnn',
    dataset_name='metrla',
    window=12,
    horizon=12,
    base_root='./data',
    results_dir='./search_results',
    min_fidelity=MIN_FIDELITY,
    max_fidelity=MAX_FIDELITY,
    total_epoch_budget=TOTAL_EPOCH_BUDGET,
    runtime_seconds=RUNTIME_SECONDS,
    retrain_epochs=RETRAIN_EPOCHS,
    retrain_best=True,
    resume_from=None,
    seed=42,
):
    """
       Runs the DEHB search, tracks the best configuration 
       and retrains it for final evaluation.
    
    """
    os.makedirs(results_dir, exist_ok=True)
    cs = get_search_space(model_name)

    results_log = []
    search_start = time.time()
    trial_count = [0]
    epochs_used = [0]

    if resume_from is not None:
        import json
        f = open(resume_from, 'r')
        results_log = json.load(f)
        f.close()
        out_path = resume_from
        trial_count[0] = len(results_log)
        record_index = 0
        while record_index < len(results_log):
            epochs_used[0] = epochs_used[0] + results_log[record_index].get('fidelity', 0)
            record_index = record_index + 1
        print("Resuming from " + resume_from)
        print("Trials already done: " + str(trial_count[0]))
        print("Epochs already used: " + str(epochs_used[0]) + "/" + str(total_epoch_budget))
    else:
        timestamp = datetime.now().strftime('%Y%m%d')
        out_path = os.path.join(
            results_dir,
            model_name + "_" + dataset_name + "_dehb_" + timestamp + ".json"
        )

    print("=" * 50)
    print("DEHB optimisation: " + model_name + " on " + dataset_name)
    print("Min fidelity: " + str(min_fidelity) + " epochs")
    print("Max fidelity: " + str(max_fidelity) + " epochs")
    print("Epoch budget: " + str(total_epoch_budget) + " epochs")
    print("=" * 50)

    def target_function(config, fidelity, **kwargs):
        """
        Evaluates a DEHB configuration at the requested number of epochs and returns
        its validation MAE and training time.

        If there are not enough epochs left, the search stops instead of partially
        training the configuration.
        """
        requested_epochs = int(fidelity)
        remaining = total_epoch_budget - epochs_used[0]

        if requested_epochs > remaining:
            print("Epoch budget exhausted: " + str(epochs_used[0]) +
                  "/" + str(total_epoch_budget) + " epochs used, " +
                  "next trial needs " + str(requested_epochs))
            raise BudgetExhausted()

        trial_count[0] = trial_count[0] + 1
        config_dict = {}
        raw_config = dict(config)
        for key in raw_config:
            config_dict[key] = to_native(raw_config[key])

        lr = config_dict.pop('lr')
        batch_size = config_dict.pop('batch_size')
        model_kwargs = config_dict

        print("Trial " + str(trial_count[0]) +
              " | Fidelity: " + str(requested_epochs) + " epochs" +
              " | Budget: " + str(epochs_used[0]) + "/" + str(total_epoch_budget) +
              " | lr=" + str(round(lr, 5)) +
              " | batch_size=" + str(batch_size))

        trial_start = time.time()
        epochs_used[0] = epochs_used[0] + requested_epochs

        try:
            predictor, trainer, test_results, best_model_path, best_val_mae = train(
                dataset_name=dataset_name,
                model_name=model_name,
                window=window,
                horizon=horizon,
                batch_size=batch_size,
                lr=lr,
                max_epochs=requested_epochs,
                base_root=base_root,
                model_kwargs=model_kwargs,
            )

            timings = dict(LAST_RUN_TIMINGS)
            val_mae = best_val_mae
            test_mae = test_results[0].get('test_mae', None)
            test_rmse = test_results[0].get('test_rmse', None)
            test_mape = test_results[0].get('test_mape', None)
            mae_15 = test_results[0].get('test_mae_at_15', None)
            mae_30 = test_results[0].get('test_mae_at_30', None)
            mae_60 = test_results[0].get('test_mae_at_60', None)

            print("  Val MAE: " + str(round(val_mae, 4)))

            trial_record = {
                'trial': trial_count[0],
                'fidelity': requested_epochs,
                'epochs_used_total': epochs_used[0],
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_mape': test_mape,
                'mae_at_15': mae_15,
                'mae_at_30': mae_30,
                'mae_at_60': mae_60,
                'best_model_path': best_model_path,
            }
            trial_record.update(timings)

        except Exception as e:
            print("  Trial " + str(trial_count[0]) + " failed: " + str(e))
            val_mae = float('inf')
            trial_record = {
                'trial': trial_count[0],
                'fidelity': requested_epochs,
                'epochs_used_total': epochs_used[0],
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'error': str(e),
            }

        trial_end = time.time()
        trial_record['trial_duration_sec'] = trial_end - trial_start
        trial_record['elapsed_since_start_sec'] = trial_end - search_start
        results_log.append(trial_record)

        save_json(results_log, out_path)

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
        seed=seed,
        output_path=os.path.join(results_dir, 'dehb_internal_' + model_name + '_' + dataset_name),
        resume=(resume_from is not None),
    )

    try:
        optimizer.run(total_cost=runtime_seconds)
    except BudgetExhausted:
        print("Search stopped: epoch budget reached.")

    session_wallclock = time.time() - search_start
    search_wallclock = sum(t.get('trial_duration_sec', 0) for t in results_log)

    print("")
    print("Epochs used: " + str(epochs_used[0]) + "/" + str(total_epoch_budget))
    print("Trials run: " + str(trial_count[0]))
    print("Wallclock (all sessions): " + str(round(search_wallclock, 1)) + "s")
    print("Wallclock (this session): " + str(round(session_wallclock, 1)) + "s")
    print("Overhead: " + str(round(total_overhead(results_log), 1)) + "s")

    best = get_incumbent(results_log)
    if best is None:
        print("No trials completed successfully.")
        return results_log

    print("Best trial: " + str(best['trial']) +
          " | Val MAE: " + str(round(best['val_mae'], 4)) +
          " | at fidelity " + str(best['fidelity']))
    print("Config: lr=" + str(best['lr']) +
          ", batch_size=" + str(best['batch_size']) +
          ", model_kwargs=" + str(best['model_kwargs']))

    if retrain_best:
        retrain_incumbent(
            best=best,
            model_name=model_name,
            dataset_name=dataset_name,
            window=window,
            horizon=horizon,
            base_root=base_root,
            results_dir=results_dir,
            retrain_epochs=retrain_epochs,
            search_epochs_used=epochs_used[0],
            search_trials=trial_count[0],
            search_wallclock_sec=search_wallclock,
            search_overhead_sec=total_overhead(results_log),
        )

    return results_log


def get_incumbent(results_log):
    """
    Returns the trial record with the lowest validation MAE, or None if no
    trial completed. The best configuration is selected using validation MAE.
    """
    valid_results = []
    record_index = 0
    while record_index < len(results_log):
        record = results_log[record_index]
        if 'error' not in record and record.get('val_mae', None) is not None:
            valid_results.append(record)
        record_index = record_index + 1

    if len(valid_results) == 0:
        return None

    return min(valid_results, key=lambda r: r['val_mae'])


def total_overhead(results_log):
    """
    Calculates the total overhead from data loading and testing across all trials
    """
    total = 0.0
    record_index = 0
    while record_index < len(results_log):
        total = total + results_log[record_index].get('overhead_sec', 0.0)
        record_index = record_index + 1
    return total


def retrain_incumbent(
    best,
    model_name,
    dataset_name,
    window,
    horizon,
    base_root,
    results_dir,
    retrain_epochs,
    search_epochs_used=None,
    search_trials=None,
    search_wallclock_sec=None,
    search_overhead_sec=None,
    search_method='dehb',
    retrain_patience=30,
):
    """
    Retrains the best configuration from scratch for final evaluation, 
    using early stopping to determine convergence
    """
    print("=" * 50)
    print("Retraining incumbent: " + model_name + " on " + dataset_name)
    print("Epochs: up to " + str(retrain_epochs) + " with early stopping")
    print("=" * 50)

    predictor, trainer, test_results, best_model_path, best_val_mae = train(
        dataset_name=dataset_name,
        model_name=model_name,
        window=window,
        horizon=horizon,
        batch_size=best['batch_size'],
        lr=best['lr'],
        max_epochs=retrain_epochs,
        base_root=base_root,
        model_kwargs=best['model_kwargs'],
        patience=retrain_patience,
    )

    timings = dict(LAST_RUN_TIMINGS)
    val_mae = best_val_mae
    test_mae = test_results[0].get('test_mae', None)
    test_rmse = test_results[0].get('test_rmse', None)
    test_mape = test_results[0].get('test_mape', None)
    mae_15 = test_results[0].get('test_mae_at_15', None)
    mae_30 = test_results[0].get('test_mae_at_30', None)
    mae_60 = test_results[0].get('test_mae_at_60', None)
    rmse_15 = test_results[0].get('test_rmse_at_15', None)
    rmse_30 = test_results[0].get('test_rmse_at_30', None)
    rmse_60 = test_results[0].get('test_rmse_at_60', None)
    mape_15 = test_results[0].get('test_mape_at_15', None)
    mape_30 = test_results[0].get('test_mape_at_30', None)
    mape_60 = test_results[0].get('test_mape_at_60', None)

    print("Retrained incumbent results for " + dataset_name + ":")
    print("  Val MAE: " + str(val_mae))
    print("  Test MAE: " + str(test_mae))
    print("  Test RMSE: " + str(test_rmse))
    print("  Test MAPE: " + str(test_mape))
    print("  MAE at 15min: " + str(mae_15))
    print("  MAE at 30min: " + str(mae_30))
    print("  MAE at 60min: " + str(mae_60))

    timestamp = datetime.now().strftime('%Y%m%d')
    out_path = os.path.join(
        results_dir,
        model_name + "_" + dataset_name + "_" + search_method + "_incumbent_" + timestamp + ".json"
    )
    record = {
        'model': model_name,
        'dataset': dataset_name,
        'phase': 2,
        'search_method': search_method,
        'search_epochs_used': search_epochs_used,
        'search_trials': search_trials,
        'search_wallclock_sec': search_wallclock_sec,
        'search_overhead_sec': search_overhead_sec,
        'search_trial': best['trial'],
        'search_fidelity': best['fidelity'],
        'retrain_epochs': retrain_epochs,
        'retrain_patience': retrain_patience,
        'lr': best['lr'],
        'batch_size': best['batch_size'],
        'model_kwargs': best['model_kwargs'],
        'search_val_mae': best['val_mae'],
        'val_mae': val_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'mae_at_15': mae_15,
        'mae_at_30': mae_30,
        'mae_at_60': mae_60,
        'rmse_at_15': rmse_15,
        'rmse_at_30': rmse_30,
        'rmse_at_60': rmse_60,
        'mape_at_15': mape_15,
        'mape_at_30': mape_30,
        'mape_at_60': mape_60,
        'best_model_path': best_model_path,
    }
    record.update(timings)
    save_json(record, out_path)

    try:
        results_to_excel(out_path)
    except Exception as e:
        print("Excel export failed: " + str(e))

    print("Incumbent results saved to " + out_path)
    return record


def main():
    datasets = ['metrla', 'pemsbay']
    dataset_index = 0
    while dataset_index < len(datasets):
        run_dehb(
            model_name='dcrnn',
            dataset_name=datasets[dataset_index],
        )
        dataset_index = dataset_index + 1


if __name__ == "__main__":
    main()
