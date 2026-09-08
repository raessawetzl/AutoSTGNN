from random_search import run_random_search
def main():
    run_random_search(
        model_name='stgcn',
        dataset_name='pemsbay',
        n_trials=25,
        max_epochs=11, 
    )
if __name__ == "__main__":
        main()