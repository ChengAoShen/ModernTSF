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


def corr(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-channel Pearson correlation between pred and true.

    Arrays are expected to be shaped ``(B, pred_len, C)``. The correlation is
    computed per channel over the flattened ``(B, pred_len)`` samples, then
    averaged across channels. Channels with zero variance contribute 0.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    c = p.shape[-1]
    p = p.reshape(-1, c)
    t = t.reshape(-1, c)
    p_mean = p.mean(axis=0, keepdims=True)
    t_mean = t.mean(axis=0, keepdims=True)
    p_dev = p - p_mean
    t_dev = t - t_mean
    num = (p_dev * t_dev).sum(axis=0)
    denom = np.sqrt((p_dev**2).sum(axis=0) * (t_dev**2).sum(axis=0))
    per_channel = np.divide(
        num, denom, out=np.zeros_like(num), where=denom > 1e-12
    )
    return float(np.nan_to_num(per_channel).mean())


def rse(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative squared error: ||pred-true||_2 / ||true-mean(true)||_2."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    num = np.sqrt(np.sum((t - p) ** 2))
    denom = np.sqrt(np.sum((t - t.mean()) ** 2))
    return float(num / (denom + 1e-5))


def wape(pred: np.ndarray, true: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|pred-true| / sum|true|."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    return float(np.sum(np.abs(p - t)) / (np.sum(np.abs(t)) + 1e-5))


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    """Symmetric mean absolute percentage error (in percent)."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    val = 2.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)
    return float(np.mean(np.nan_to_num(val)) * 100.0)


def mase(pred: np.ndarray, true: np.ndarray, season: int = 1) -> float:
    """Mean absolute scaled error.

    MASE = mean|pred-true| / mean|naive seasonal error|.

    The seasonal naive error is computed over the concatenated ground-truth
    sequence (flattened across the batch and horizon dimensions) using a lag of
    ``season`` steps (default 1 = last-value naive).

    Limitation: this benchmark only has the prediction window available at
    evaluation time, not the full historical series. The naive baseline is
    therefore derived from the test targets themselves rather than from the
    in-sample training history, so the absolute MASE values are not directly
    comparable to the textbook in-sample-scaled definition. If the denominator
    cannot be computed (too few samples or zero seasonal error) the function
    returns ``float('nan')``.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    season = max(int(season), 1)
    flat = t.reshape(-1)
    if flat.shape[0] <= season:
        return float("nan")
    naive = np.mean(np.abs(flat[season:] - flat[:-season]))
    if not np.isfinite(naive) or naive <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(p - t)) / naive)


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
        "corr": corr(pred, true),
        "rse": rse(pred, true),
        "wape": wape(pred, true),
        "smape": smape(pred, true),
        "mase": mase(pred, true),
    }


def register() -> None:
    """Register built-in metrics into the registry."""
    METRIC_REGISTRY.register("mae", mae)
    METRIC_REGISTRY.register("mse", mse)
    METRIC_REGISTRY.register("rmse", rmse)
    METRIC_REGISTRY.register("mape", mape)
    METRIC_REGISTRY.register("mspe", mspe)
    METRIC_REGISTRY.register("corr", corr)
    METRIC_REGISTRY.register("rse", rse)
    METRIC_REGISTRY.register("wape", wape)
    METRIC_REGISTRY.register("smape", smape)
    METRIC_REGISTRY.register("mase", mase)
