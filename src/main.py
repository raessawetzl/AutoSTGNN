from random_search import run_random_search
def main():
    run_random_search(
        model_name='stgcn',
        dataset_name='MetrLA',
        n_trials=20,
        max_epochs=15,
    )
if __name__ == "__main__":
        main()