import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics_regression(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Computes the regression performance metrics

    Args:
        pred (np.ndarray): The prediction vector
        gt (np.ndarray): The ground truth vector

    Returns:
        tuple[float | None, float | None, float | None]: R^2, MAE, MSE
    """
    r2 = r2_score(gt, pred) if len(gt) > 0 else None
    mae = mean_absolute_error(gt, pred) if len(gt) > 0 else None
    mse = mean_squared_error(gt, pred) if len(gt) > 0 else None
    return (r2, mae, mse)


def compute_metrics_classification(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[float | None, float | None]:
    """Computes the classification performance metrics

    Args:
        pred (np.ndarray): The prediction vector
        gt (np.ndarray): The ground truth vector

    Returns:
        tuple[float | None, float | None]: Accuracy, F1
    """
    acc = (
        accuracy_score(gt, pred)
        if gt is not None and pred is not None and len(gt) > 0
        else None
    )
    f1 = (
        f1_score(gt, pred, pos_label=1)
        if gt is not None and pred is not None and len(gt) > 0
        else None
    )
    return (acc, f1)
