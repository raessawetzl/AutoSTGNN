from BODE import run_bode
def main():
    run_bode(
        model_name='graphwavenet',
        dataset_name='metrla',
        T=10,
        n_init=5,
        max_epochs=10,
    )
if __name__ == "__main__":
        main()
