"""Regression metrics shared by all tabular baselines."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score


def to_numpy_1d(value):
    """Convert tensors/lists/arrays into flat NumPy arrays."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def regression_metrics(y_true, y_pred):
    """Compute MSE, RMSE, NMSE, and R² for regression."""
    y_true = to_numpy_1d(y_true)
    y_pred = to_numpy_1d(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    variance = float(np.mean((y_true - np.mean(y_true)) ** 2))
    nmse = float(mse / variance) if variance > 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "nmse": nmse,
        "r2": r2,
    }
