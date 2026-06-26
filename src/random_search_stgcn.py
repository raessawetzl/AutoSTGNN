import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from search_space import get_stgcn_config_space
from train_stgcn import train

def random_search(n_configs=20, epochs=10, dataset_name="PEMS04"):
    cs = get_stgcn_config_space()

    results = []

    for i in range(n_configs):
        # sample a random config from the search space
        config = cs.sample_configuration()
        config_dict = {k: int(v) if isinstance(v, np.integer) else v for k, v in dict(config).items()}

        print(f"\nconfig {i+1}/{n_configs}: {config_dict}")

        val_mae = train(config_dict, dataset_name=dataset_name, epochs=epochs)
        results.append((val_mae, config_dict))
        print(f"val MAE: {val_mae:.4f}")

    # sort by best val MAE
    results.sort(key=lambda x: x[0])

    print("\n--- top 5 configs ---")
    for val_mae, config_dict in results[:5]:
        print(f"val MAE: {val_mae:.4f} | {config_dict}")

    return results

if __name__ == "__main__":
    random_search(n_configs=20, epochs=10, dataset_name="PEMS04")