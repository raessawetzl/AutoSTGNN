import torch


def mae(pred, real, mask=None):
    if mask is not None:
        pred, real = pred[mask], real[mask]
    return torch.mean(torch.abs(pred - real)).item()


def rmse(pred, real, mask=None):
    if mask is not None:
        pred, real = pred[mask], real[mask]
    return torch.sqrt(torch.mean((pred - real) ** 2)).item()


def mape(pred, real, mask=None, threshold=1e-3):
    if mask is not None:
        pred, real = pred[mask], real[mask]
    # extra safety: even within masked (valid) points, avoid divide-by-near-zero
    safe = real.abs() > threshold
    if safe.sum() == 0:
        return float('nan')
    return torch.mean(torch.abs((pred[safe] - real[safe]) / real[safe]) * 100).item()


def compute_metrics(pred, real, mask=None):
    return {
        "MAE": mae(pred, real, mask),
        "RMSE": rmse(pred, real, mask),
        "MAPE": mape(pred, real, mask),
    }


def evaluate_per_horizon(pred, real, horizons=(3, 6, 12), mask=None):
    results = {}
    for h in horizons:
        step_pred = pred[:, h - 1]
        step_real = real[:, h - 1]
        step_mask = mask[:, h - 1] if mask is not None else None
        results[h] = compute_metrics(step_pred, step_real, step_mask)
    return results