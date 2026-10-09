from typing import Any, Optional

import numpy as np
from hdbscan import HDBSCAN  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)


def compute_metrics_clustering(
    data: np.ndarray, labels: list[int], model: Optional[HDBSCAN]
) -> dict[str, Any]:
    _labels = np.asarray(labels)
    mask = _labels != -1
    n_clusters = len(set(_labels[mask]))

    metrics: dict = {
        "n_jobs": len(data),
        "n_clusters": n_clusters,
        "noise_ratio": float(np.mean(labels == -1)),
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "dbcv": None,
        "cluster_persistence": None,
    }

    # These require at least 2 clusters and some non-noise points
    if n_clusters > 1 and mask.sum() > n_clusters:
        data_clean = data[mask]
        y_clean = labels[mask]
        metrics["silhouette"] = float(silhouette_score(data_clean, y_clean))
        metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(data_clean, y_clean)
        )
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(data_clean, y_clean)
        )

    if model is not None and hasattr(model, "relative_validity_"):
        metrics["dbcv"] = float(model.relative_validity_)

    if model is not None and hasattr(model, "cluster_persistence_"):
        persistence = np.asarray(model.cluster_persistence_)
        if persistence.size > 0:
            metrics["cluster_persistence"] = float(persistence.mean())

    return metrics


def compute_metrics_regression(
    pred: list[float], gt: list[float]
) -> tuple[float | None, float | None, float | None]:
    """Computes the regression performance metrics

    Args:
        pred (list[float]): The prediction vector
        gt (list[float]): The ground truth vector

    Returns:
        tuple[float | None, float | None, float | None]: R^2, MAE, MSE
    """
    r2 = r2_score(gt, pred) if len(gt) > 0 else None
    mae = mean_absolute_error(gt, pred) if len(gt) > 0 else None
    mse = mean_squared_error(gt, pred) if len(gt) > 0 else None
    return (r2, mae, mse)


def compute_metrics_classification(
    pred: list[float], gt: list[float]
) -> tuple[float | None, float | None]:
    """Computes the classification performance metrics

    Args:
        pred (list[float]): The prediction vector
        gt (list[float]): The ground truth vector

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
