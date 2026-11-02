"""Standard regression metrics for quantum-circuit runtime prediction.

"""

# Import NumPy for array conversion and numerical metric calculations.
import numpy as np

# Import Torch because predictions and targets often arrive as tensors.
import torch

from sklearn.metrics import r2_score, mean_squared_error


def to_numpy_1d(value):
    """Convert a tensor/list/array into a flat NumPy array.
    """
    # If the input is a PyTorch tensor, detach it from the computation graph.
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = value.numpy()
    return np.asarray(value).reshape(-1)


def regression_metrics(y_true, y_pred):
    """Compute MSE, RMSE, NMSE, and R² for regression.

    y_true is the measured simulator runtime.
    y_pred is the model-predicted simulator runtime.
    """
    y_true = to_numpy_1d(y_true)
    y_pred = to_numpy_1d(y_pred)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    
    r2 = float(r2_score(y_true, y_pred))
    
    variance = float(np.mean((y_true - np.mean(y_true)) ** 2))
    nmse = float(mse / variance) if variance > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "nmse": nmse, "r2": r2}
