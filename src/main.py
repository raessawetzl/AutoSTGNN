from random_search import run_random_search
def main():
    run_random_search(
        model_name='dcrnn',
        dataset_name='pemsbay',
        n_trials=10,
        max_epochs=10,
    )
if __name__ == "__main__":
        main()