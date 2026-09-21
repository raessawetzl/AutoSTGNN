"""
trainer.py

model construction and training loop for stgnn models.

wraps tsl model classes with default hyperparameters, builds a Predictor
(loss, optimizer, metrics), and runs a pytorch-lightning training loop with
early stopping and best-checkpoint saving. also patches torch.load to trust
lightning checkpoints (weights_only=False), since tsl models aren't plain
tensors.

usage:
    from trainer import train
    predictor, trainer, test_results, best_model_path, best_val_mae = train(
        dataset_name='electricity', model_name='stgcn', max_epochs=30,
    )

config:
    dataset_name    - dataset to train/eval on (metrla, pemsbay, pems04, pems08, electricity)
    model_name      - model to train (graphwavenet, dcrnn, stgcn, agcrn)
    window          - input window length
    horizon         - forecast horizon length
    batch_size      - dataloader batch size
    lr              - learning rate
    max_epochs      - max training epochs
    base_root       - root directory containing the dataset
    model_kwargs    - overrides for the model's default hyperparameters
    checkpoint_dir  - root directory for saved checkpoints
    save_best       - whether to save/restore the best-val checkpoint
    seed            - random seed
    horizon_steps   - forecast steps to report per-step metrics at
    patience        - early-stopping patience (epochs)
"""

import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from tsl.nn.models import GraphWaveNetModel, DCRNNModel, STCNModel, AGCRNModel
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataloader import get_dataloaders


_torch_load = torch.load


def _trusted_torch_load(*args, **kwargs):
    """torch.load wrapper that trusts our own checkpoints (weights_only=False)."""
    kwargs['weights_only'] = False
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


MODEL_MAP = {
    'graphwavenet': GraphWaveNetModel,
    'dcrnn': DCRNNModel,
    'stgcn': STCNModel,
    'agcrn': AGCRNModel,
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
    """seed python, numpy, torch (incl. cuda) and lightning for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def get_model(model_name, n_nodes, input_size, output_size, horizon, model_kwargs=None):
    """build a model instance from MODEL_MAP, merging model_kwargs onto its defaults.

    returns (model, used_kwargs) where used_kwargs is the final kwargs dict
    the model was actually constructed with.
    """
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

    # these models need the node count to build a learned adjacency/embedding
    if model_name == 'graphwavenet' and kwargs.get('learned_adjacency', True):
        kwargs['n_nodes'] = n_nodes

    if model_name == 'agcrn':
        kwargs['n_nodes'] = n_nodes

    model = model_cls(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        **kwargs,
    )

    return model, kwargs


class MaskedRMSE(MaskedMSE):
    """masked rmse metric, built on top of tsl's MaskedMSE."""

    def compute(self):
        return torch.sqrt(super().compute())


def build_metrics(horizon_steps=(3, 6, 12)):
    """build the mae/mape/rmse metric dict, overall and at each step in horizon_steps.

    horizon_steps: forecast steps (1-indexed) to report metrics at, e.g.
    (3, 6, 12) hours for hourly-sampled data, or (3, 6, 12) representing
    15/30/60 min for 5-min-sampled traffic data. tsl's `at` is 0-indexed,
    so step N corresponds to at=N-1.
    """
    metrics = {
        'mae': MaskedMAE(),
        'mape': MaskedMAPE(),
        'rmse': MaskedRMSE(),
    }

    for step in horizon_steps:
        label = str(step)
        at = step - 1
        metrics[f'mae_at_{label}'] = MaskedMAE(at=at)
        metrics[f'mape_at_{label}'] = MaskedMAPE(at=at)
        metrics[f'rmse_at_{label}'] = MaskedRMSE(at=at)

    return metrics


def train(
    dataset_name='metrla',    # options: metrla, pemsbay, pems04, pems08, electricity
    model_name='dcrnn',       # options: graphwavenet, dcrnn, stgcn, agcrn
    window=None,
    horizon=12,
    batch_size=64,
    lr=1e-3,
    max_epochs=50,
    base_root='./data',
    model_kwargs=None,
    checkpoint_dir='./checkpoints',
    save_best=True,
    seed=42,
    horizon_steps=(3, 6, 12),
    patience=30,
):
    """train and test a model end-to-end.

    returns (predictor, trainer, test_results, best_model_path, best_val_mae).
    """
    set_seed(seed)
    torch.set_float32_matmul_precision('medium')

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        base_root=base_root,
    )

    # infer model I/O dims directly from one batch instead of hardcoding them
    sample_batch = next(iter(train_loader))
    n_nodes = sample_batch.input.x.shape[2]
    input_size = sample_batch.input.x.shape[-1]
    output_size = sample_batch.target.y.shape[-1]

    # auto-detect exogenous size from the batch (e.g. mask_as_exog on AQI),
    # so this works whether or not 'u' is present
    exog_size = sample_batch.input.u.shape[-1] if 'u' in sample_batch.input else 0
    model_kwargs = dict(model_kwargs) if model_kwargs else {}
    model_kwargs.setdefault('exog_size', exog_size)

    model, used_kwargs = get_model(
        model_name=model_name,
        n_nodes=n_nodes,
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        model_kwargs=model_kwargs,
    )
    print(f"Training {model_name} on {dataset_name} with: {used_kwargs}")

    loss_fn = MaskedMAE()
    metrics = build_metrics(horizon_steps=horizon_steps)

    predictor = Predictor(
        model=model,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': lr},
        loss_fn=loss_fn,
        metrics=metrics,
    )

    callbacks = [EarlyStopping(monitor='val_mae', patience=patience, mode='min')]

    checkpoint_callback = None
    if save_best:
        run_dir = os.path.join(checkpoint_dir, dataset_name, model_name)
        os.makedirs(run_dir, exist_ok=True)

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

    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None
    test_results = trainer.test(predictor, dataloaders=test_loader, ckpt_path=best_model_path)

    best_val_mae = (
        float(checkpoint_callback.best_model_score)
        if checkpoint_callback and checkpoint_callback.best_model_score is not None
        else None
    )
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")

    return predictor, trainer, test_results, best_model_path, best_val_mae