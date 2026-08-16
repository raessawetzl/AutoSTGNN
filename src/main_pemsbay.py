from random_search import run_random_search
def main():
    run_random_search(
        model_name='stgcn',
        dataset_name='pemsbay',
        n_trials=15,
        max_epochs=20, 
    )
if __name__ == "__main__":
        main()