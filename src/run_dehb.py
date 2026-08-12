from dehb_runner import run_dehb

def main():
    run_dehb(
        model_name='dcrnn',
        dataset_name='metrla',
        min_fidelity=3,
        max_fidelity=15,
        runtime_seconds=7200,
    )

if __name__ == "__main__":
    main()