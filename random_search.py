import json
import os
import numpy as np
from datetime import datetime

from search_space import get_search_space
from trainer import train


def to_native(value):
    """Convert numpy scalar types to native Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def run_random_search(
    model_name='dcrnn',
    dataset_name='metrla',
    n_trials=20,
    max_epochs=20,
    window=12,
    horizon=12,
    base_root='./data',
    results_dir='./search_results',
):
    os.makedirs(results_dir, exist_ok=True)

    cs = get_search_space(model_name)
    configs = cs.sample_configuration(n_trials)
    if n_trials == 1:
        configs = [configs]

    results_log = []

    for i, config in enumerate(configs):
        config_dict = {k: to_native(v) for k, v in dict(config).items()}

        lr = config_dict.pop('lr')
        batch_size = config_dict.pop('batch_size')
        model_kwargs = config_dict

        print(f"\n=== Trial {i+1}/{n_trials} ===")
        print(f"lr={lr}, batch_size={batch_size}, model_kwargs={model_kwargs}")

        try:
            predictor, trainer, test_results = train(
                dataset_name=dataset_name,
                model_name=model_name,
                window=window,
                horizon=horizon,
                batch_size=batch_size,
                lr=lr,
                max_epochs=max_epochs,
                base_root=base_root,
                model_kwargs=model_kwargs,
            )

            test_mae = test_results[0].get('test_mae', None)

            trial_record = {
                'trial': i,
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'test_mae': test_mae,
                'test_results': test_results[0],
            }

        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            trial_record = {
                'trial': i,
                'lr': lr,
                'batch_size': batch_size,
                'model_kwargs': model_kwargs,
                'error': str(e),
            }

        results_log.append(trial_record)

        timestamp = datetime.now().strftime('%Y%m%d')
        out_path = os.path.join(
            results_dir, f"{model_name}_{dataset_name}_{timestamp}.json"
        )
        with open(out_path, 'w') as f:
            json.dump(results_log, f, indent=2)

    valid_results = [r for r in results_log if 'test_mae' in r]
    if valid_results:
        best = min(valid_results, key=lambda r: r['test_mae'])
        print(f"\nBest trial: {best['trial']} with test_mae={best['test_mae']}")
        print(f"Config: lr={best['lr']}, batch_size={best['batch_size']}, model_kwargs={best['model_kwargs']}")
    else:
        print("\nNo trials completed successfully.")

    return results_log


if __name__ == '__main__':
    run_random_search(
        model_name='dcrnn',
        dataset_name='metrla',
        n_trials=20,
        max_epochs=20,
    )