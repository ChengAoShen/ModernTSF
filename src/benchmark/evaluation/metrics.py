"""Built-in metric implementations and registration."""

from __future__ import annotations

import numpy as np

from benchmark.registry import METRIC_REGISTRY


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(true - pred)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((true - pred) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute percentage error."""
    return float(np.mean(np.abs((true - pred) / true)))


def mspe(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared percentage error."""
    return float(np.mean(np.square((true - pred) / true)))


def collect_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Compute the default metric suite.

    Parameters
    ----------
    pred : np.ndarray
        Model predictions.
    true : np.ndarray
        Ground-truth targets.

    Returns
    -------
    dict[str, float]
        Metrics keyed by name.
    """
    return {
        "mae": mae(pred, true),
        "mse": mse(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
        "mspe": mspe(pred, true),
    }


def register() -> None:
    """Register built-in metrics into the registry."""
    METRIC_REGISTRY.register("mae", mae)
    METRIC_REGISTRY.register("mse", mse)
    METRIC_REGISTRY.register("rmse", rmse)
    METRIC_REGISTRY.register("mape", mape)
    METRIC_REGISTRY.register("mspe", mspe)
