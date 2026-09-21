"""
main.py

entry point script for hyperparameter search.

runs a random search over stgnn model hyperparameters on a certain dataset,
using run_random_search from random_search.py to handle trial sampling,
training, and evaluation for each configuration.

usage:
    python main.py

config:
    model_name   - model to search over 
    dataset_name - dataset to train/eval on 
    n_trials     - number of random configs to try
    max_epochs   - max training epochs per trial
"""

from random_search import run_random_search
def main():
    run_random_search(
        model_name='stgcn', # options: stgcn, graphwavenet, agcrn, dcrnn
        dataset_name='electricity', # options: metrla, pemsbay, electricity, pems04, pems08 
        n_trials=20, 
        max_epochs=12, 
    )
if __name__ == "__main__":
        main()