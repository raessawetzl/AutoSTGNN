import os
from datetime import datetime

from trainer import train, DEFAULT_MODEL_KWARGS, LAST_RUN_TIMINGS
from utils import get_best_val_mae, save_json, results_to_excel

# Trains AGCRN, Graph WaveNet and STCN using their default hyperparameters
# across METR-LA, PEMS-BAY and Electricity to provide the untuned baselines
# The same training pipeline is used for the baseline and tuned configurations

MODELS = ['agcrn', 'graphwavenet', 'stgcn']
DATASETS = ['metrla', 'pemsbay', 'electricity']

WINDOW = 12          # number of historical timesteps fed into the model (1 hour)
HORIZON = 12         # number of future timesteps predicted (1 hour)
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MAX_EPOCHS = 100     # Maximum training epochs with early stopping
BASE_ROOT = './data'
RESULTS_DIR = './baseline_results'


def run_baseline(model_name, dataset_name):
    """Trains one model with its default configuration and records its performance"""
    print("=" * 50)
    print("Baseline: " + model_name + " on " + dataset_name)
    print("=" * 50)

    predictor, trainer, test_results, best_model_path = train(
        dataset_name=dataset_name,
        model_name=model_name,
        window=WINDOW,
        horizon=HORIZON,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        base_root=BASE_ROOT,
        model_kwargs=None,
        checkpoint_dir='./checkpoints_baseline',
    )

    timings = dict(LAST_RUN_TIMINGS)
    val_mae = get_best_val_mae(trainer)

    record = {
        'model': model_name,
        'dataset': dataset_name,
        'tuned': False,
        'search_method': 'none',
        'search_epochs_used': 0,
        'search_trials': 0,
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'model_kwargs': dict(DEFAULT_MODEL_KWARGS[model_name]),
        'val_mae': val_mae,
        'test_mae': test_results[0].get('test_mae', None),
        'test_mape': test_results[0].get('test_mape', None),
        'mae_at_15': test_results[0].get('test_mae_at_15', None),
        'mae_at_30': test_results[0].get('test_mae_at_30', None),
        'mae_at_60': test_results[0].get('test_mae_at_60', None),
        'best_model_path': best_model_path,
    }
    record.update(timings)

    print("Val MAE: " + str(record['val_mae']))
    print("Test MAE: " + str(record['test_mae']))
    print("Test MAPE: " + str(record['test_mape']))

    return record


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')
    out_path = os.path.join(RESULTS_DIR, "baselines_" + timestamp + ".json")

    results = []

    dataset_index = 0
    while dataset_index < len(DATASETS):
        dataset_name = DATASETS[dataset_index]

        model_index = 0
        while model_index < len(MODELS):
            model_name = MODELS[model_index]

            try:
                record = run_baseline(model_name, dataset_name)
            except Exception as e:
                print("Baseline failed for " + model_name + " on " + dataset_name + ": " + str(e))
                record = {
                    'model': model_name,
                    'dataset': dataset_name,
                    'tuned': False,
                    'error': str(e),
                }

            results.append(record)
            save_json(results, out_path)

            try:
                results_to_excel(out_path)
            except Exception as e:
                print("Excel export failed: " + str(e))

            model_index = model_index + 1

        dataset_index = dataset_index + 1

    print("")
    print("=" * 50)
    print("Baseline summary")
    print("=" * 50)
    print("{:<15} {:<10} {:<10} {:<10}".format("Model", "Dataset", "Val MAE", "Test MAE"))

    result_index = 0
    while result_index < len(results):
        r = results[result_index]
        if 'error' in r:
            print("{:<15} {:<10} {:<10} {:<10}".format(
                r['model'], r['dataset'], "failed", "failed"))
        else:
            print("{:<15} {:<10} {:<10} {:<10}".format(
                r['model'],
                r['dataset'],
                round(r['val_mae'], 4),
                round(r['test_mae'], 4) if r['test_mae'] is not None else "n/a"
            ))
        result_index = result_index + 1

    print("Baseline results saved to " + out_path)


if __name__ == "__main__":
    main()
