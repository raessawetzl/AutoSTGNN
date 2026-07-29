from trainer import train
def main():
        predictor, trainer, results = train(
        dataset_name='metrla',
        model_name='graphwavenet',
        model_kwargs={'hidden_size': 128, 'n_layers': 2}
    )
if __name__ == "__main__":
        main()