import os
import time
import random
import numpy as np
import torch
import pytorch_lightning as pl
from tsl.nn.models import GraphWaveNetModel, DCRNNModel, STCNModel, AGCRNModel
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataloader import get_dataloaders

import torch.serialization

_orig_torch_load = torch.load
def _torch_load_full(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_full

# Maps each model name to its corresponding TSL implementation
MODEL_MAP = {
    'graphwavenet': GraphWaveNetModel,
    'dcrnn': DCRNNModel,
    'stgcn': STCNModel,
    'agcrn': AGCRNModel
}

# Default hyperparameters used for the untuned baseline models
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
        'exog_size': 0,
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


def set_seed(seed=42):
    """Sets the random seeds used for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def get_model(model_name, n_nodes, input_size, output_size, horizon, model_kwargs=None):
    """Creates a model using its default parameters and any supplied hyperparameters."""
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

    # Graph WaveNet requires the number of nodes when learning the adjacency matrix
    if model_name == 'graphwavenet' and kwargs.get('learned_adjacency', True):
        kwargs['n_nodes'] = n_nodes

    # AGCRN requires the number of nodes for its adaptive graph
    if model_name == 'agcrn':
        kwargs['n_nodes'] = n_nodes

    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        **kwargs
    )

    return model, kwargs

LAST_RUN_TIMINGS = {}


class MaskedRMSE(MaskedMSE):
    """Calculates RMSE while ignoring masked values."""
    def compute(self):
        return torch.sqrt(super().compute())


def build_metrics():
    """Creates the evaluation metrics for the overall and forecast horizon results"""
    HORIZON_POINTS = {
        '15': 2,
        '30': 5,
        '60': 11,
    }

    metrics = {
        'mae': MaskedMAE(),
        'mape': MaskedMAPE(),
        'rmse': MaskedRMSE(),
    }

    for label, step in HORIZON_POINTS.items():
        metrics[f'mae_at_{label}'] = MaskedMAE(at=step)
        metrics[f'mape_at_{label}'] = MaskedMAPE(at=step)
        metrics[f'rmse_at_{label}'] = MaskedRMSE(at=step)

    return metrics


def train(
    dataset_name='metrla',
    model_name='agcrn',
    window=12,
    horizon=12,
    batch_size=64,
    lr=1e-3,
    max_epochs=50,
    base_root='./data',
    model_kwargs=None,
    checkpoint_dir='./checkpoints',
    patience=5,
    save_best=True,
    seed=42,
):
    """Trains and evaluates one model configuration"""
    set_seed(seed)
    torch.set_float32_matmul_precision('medium')

    setup_start = time.time()

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        base_root=base_root,
    )

    # The model dimensions are taken directly from a training batch
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
    # MAE is used as the training loss and validation metric for model selection
    loss_fn = MaskedMAE()
    metrics = build_metrics()

    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': lr},
        loss_fn=loss_fn,
        metrics=metrics
    )

    # Training stops when validation MAE stops improving
    callbacks = [EarlyStopping(monitor='val_mae', patience=patience, mode='min')]

    checkpoint_callback = None
    if save_best:
        run_dir = os.path.join(checkpoint_dir, dataset_name, model_name)
        os.makedirs(run_dir, exist_ok=True)

        # The checkpoint with the lowest validation MAE is saved
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir,
            filename='best-{epoch:02d}-{val_mae:.4f}',
            monitor='val_mae',
            mode='min',
            save_top_k=1,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
    )

    setup_end = time.time()

    fit_start = time.time()
    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
    fit_end = time.time()

    test_start = time.time()
    # The best checkpoint is used for final test evaluation
    test_results = trainer.test(predictor, dataloaders=test_loader, ckpt_path='best')
    test_end = time.time()

    # Runtime information is recorded for each training run
    LAST_RUN_TIMINGS.clear()
    LAST_RUN_TIMINGS.update({
        'setup_sec': setup_end - setup_start,
        'fit_sec': fit_end - fit_start,
        'test_sec': test_end - test_start,
        'overhead_sec': (setup_end - setup_start) + (test_end - test_start),
        'epochs_completed': trainer.current_epoch,
    })

    best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None
    best_val_mae = (
        float(checkpoint_callback.best_model_score)
        if checkpoint_callback and checkpoint_callback.best_model_score is not None
        else None
    )
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")

    return predictor, trainer, test_results, best_model_path, best_val_mae