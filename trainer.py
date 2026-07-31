import torch
import pytorch_lightning as pl
from tsl.nn.models import GraphWaveNetModel, DCRNNModel, STCNModel, AGCRNModel
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE

from dataloader import get_dataloaders


MODEL_MAP = {
    'graphwavenet': GraphWaveNetModel,
    'dcrnn': DCRNNModel,
    'stgcn': STCNModel,
    'agcrn': AGCRNModel
}

DEFAULT_MODEL_KWARGS = {
    'graphwavenet': {
        'exog_size': 0,
        'hidden_size': 32,
        'ff_size': 256,
        'n_layers': 8,
        'temporal_kernel_size': 2,
        'spatial_kernel_size': 2,
        'learned_adjacency': True,
        'emb_size': 10,
        'dilation': 2,
        'dilation_mod': 2,
        'norm': 'batch',
        'dropout': 0.3,
    },
    'dcrnn': {
        'exog_size': 0,
        'hidden_size': 32,
        'kernel_size': 2,
        'ff_size': 256,
        'n_layers': 1,
        'dropout': 0,
        'activation': 'relu',
    },
    'stgcn': {
        'hidden_size': 64,
        'ff_size': 128,
        'n_layers': 1,
        'temporal_kernel_size': 3,
        'spatial_kernel_size': 2,
        'dropout': 0.3,
    },
    'agcrn': {
        'hidden_size': 64,
        'emb_size': 10,
        'n_layers': 1,
    },
}


def get_model(model_name, n_nodes, input_size, output_size, horizon, model_kwargs=None):
    model_name = model_name.lower()
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_MAP.keys())}"
        )

    model_cls = MODEL_MAP[model_name]

    kwargs = dict(DEFAULT_MODEL_KWARGS[model_name])
    if model_kwargs:
        kwargs.update(model_kwargs)

    if model_name == 'graphwavenet' and kwargs.get('learned_adjacency', True):
        kwargs['n_nodes'] = n_nodes

    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        **kwargs
    )

    return model, kwargs


def train(
    dataset_name='metrla',
    model_name='dcrnn',
    window=12,
    horizon=12,
    batch_size=64,
    lr=1e-3,
    max_epochs=50,
    base_root='./data',
    model_kwargs=None,
):
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        base_root=base_root
    )

    sample_batch = next(iter(train_loader))
    n_nodes = sample_batch.input.x.shape[2]
    input_size = sample_batch.input.x.shape[-1]
    output_size = sample_batch.target.y.shape[-1]

    model, used_kwargs = get_model(
        model_name=model_name,
        n_nodes=n_nodes,
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        model_kwargs=model_kwargs
    )
    print(f"Training {model_name} with: {used_kwargs}")

    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': lr},
        loss_fn=MaskedMAE(),
        metrics={'mae': MaskedMAE(), 'mape': MaskedMAPE()}
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
    )

    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = trainer.test(predictor, dataloaders=test_loader)

    return predictor, trainer, test_results
