import numpy as np
import torch


def mae(pred, real):
    return torch.mean(torch.abs(pred - real)).item()


def rmse(pred, real):
    return torch.sqrt(torch.mean((pred - real) ** 2)).item()


def mape(pred, real, epsilon=1e-8):
    return torch.mean(torch.abs((pred - real) / (real + epsilon)) * 100).item()


def compute_metrics(pred, real):
    """
    Returns MAE, RMSE, MAPE for a batch of predictions.
    pred, real: torch tensors of the same shape
    """
    return {
        "MAE": mae(pred, real),
        "RMSE": rmse(pred, real),
        "MAPE": mape(pred, real)
    }