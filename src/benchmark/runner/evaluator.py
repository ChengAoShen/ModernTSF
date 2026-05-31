"""Evaluation utilities for time-series forecasting models."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from benchmark.runner.trainer import (
    _call_model,
    _make_decoder_input,
    _slice_pred_target,
)
from benchmark.evaluation.metrics import collect_metrics


def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    label_len: int,
    pred_len: int,
    features: str,
    inverse: bool = False,
    dataset=None,
) -> tuple[dict[str, float], float]:
    """Run model inference and compute metrics on a dataset split.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    data_loader : DataLoader
        Data loader for evaluation split.
    device : torch.device
        Target device.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").
    inverse : bool, optional
        Whether to inverse-transform outputs via dataset scaler.
    dataset : object, optional
        Dataset instance that provides inverse_transform.

    Returns
    -------
    tuple[dict[str, float], float]
        Metrics dictionary and evaluation time in seconds.
    """
    preds = []
    trues = []

    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = _slice_pred_target(
                outputs, batch_y, pred_len, features
            )

            outputs = outputs.detach().cpu().numpy()
            batch_y_sliced = batch_y_sliced.detach().cpu().numpy()

            if inverse and dataset is not None:
                shape = batch_y_sliced.shape
                outputs = dataset.inverse_transform(
                    outputs.reshape(shape[0] * shape[1], -1)
                ).reshape(shape)
                batch_y_sliced = dataset.inverse_transform(
                    batch_y_sliced.reshape(shape[0] * shape[1], -1)
                ).reshape(shape)

            preds.append(outputs)
            trues.append(batch_y_sliced)

    test_time = time.perf_counter() - start_time
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = collect_metrics(preds, trues)
    return metrics, test_time


def evaluate_rolling(
    model: nn.Module,
    dataset,
    device: torch.device,
    seq_len: int,
    label_len: int,
    pred_len: int,
    features: str,
    inverse: bool = False,
    horizon: int | None = None,
    stride: int = 1,
    num_rollings: int | None = None,
) -> tuple[dict[str, float], float]:
    """Run a TFB-style rolling forecast over the test split.

    Starting at the beginning of the test split, the model repeatedly consumes
    a ``seq_len`` input window, predicts ``pred_len`` steps (sliced to
    ``horizon`` if requested), then the window is advanced by ``stride`` and the
    process repeats for up to ``num_rollings`` rollings (or until the data is
    exhausted). Predictions and ground-truth targets are collected and scored
    with the same :func:`collect_metrics` used by the fixed evaluator.

    The dataset is consumed via its underlying ``data`` / ``time_stamp`` arrays
    (the materialised test split). This keeps the ``(input_series,
    output_series, input_stamp, output_stamp)`` contract intact: the input
    window plays the role of ``input_series`` (+ ``input_stamp``) and the
    label-prefixed future window plays the role of ``output_series``
    (+ ``output_stamp``).

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    dataset : object
        Test-split dataset instance exposing ``data`` (``(T, C)``), an optional
        ``time_stamp`` (``(T, M)``), and ``inverse_transform``.
    device : torch.device
        Target device.
    seq_len, label_len, pred_len : int
        Encoder input, decoder label, and prediction horizon lengths.
    features : str
        Feature mode ("M", "S", "MS").
    inverse : bool, optional
        Whether to inverse-transform outputs via the dataset scaler.
    horizon : int | None, optional
        Number of predicted steps to score per rolling. Defaults to
        ``pred_len`` and is clamped to ``pred_len`` (the model only emits
        ``pred_len`` steps per call).
    stride : int, optional
        Number of steps to advance the input window between rollings.
    num_rollings : int | None, optional
        Maximum number of rollings. ``None`` (default) rolls until the test
        data is exhausted.

    Returns
    -------
    tuple[dict[str, float], float]
        Metrics dictionary and evaluation time in seconds.
    """
    data = np.asarray(dataset.data)
    time_stamp = getattr(dataset, "time_stamp", None)

    eff_horizon = pred_len if horizon is None else min(int(horizon), pred_len)
    if eff_horizon <= 0:
        raise ValueError(f"rolling horizon must be positive, got {horizon!r}")
    if stride < 1:
        raise ValueError(f"rolling stride must be >= 1, got {stride!r}")

    total_len = data.shape[0]
    # Each rolling needs seq_len input rows plus pred_len future rows.
    last_start = total_len - seq_len - pred_len
    if last_start < 0:
        raise ValueError(
            "rolling forecast needs at least seq_len + pred_len rows in the "
            f"test split, got {total_len} for seq_len={seq_len}, "
            f"pred_len={pred_len}"
        )

    starts = list(range(0, last_start + 1, stride))
    if num_rollings is not None:
        starts = starts[: max(int(num_rollings), 0)]

    preds = []
    trues = []

    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in starts:
            input_end = start + seq_len
            output_start = input_end - label_len
            output_end = input_end + pred_len

            input_series = data[start:input_end]
            output_series = data[output_start:output_end]

            batch_x = (
                torch.from_numpy(np.ascontiguousarray(input_series))
                .float()
                .unsqueeze(0)
                .to(device)
            )
            batch_y = (
                torch.from_numpy(np.ascontiguousarray(output_series))
                .float()
                .unsqueeze(0)
                .to(device)
            )

            if time_stamp is not None:
                batch_x_mark = (
                    torch.from_numpy(np.ascontiguousarray(time_stamp[start:input_end]))
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                batch_y_mark = (
                    torch.from_numpy(
                        np.ascontiguousarray(time_stamp[output_start:output_end])
                    )
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                batch_x_mark = (
                    torch.zeros((1, seq_len, 6), dtype=torch.float32).to(device)
                )
                batch_y_mark = (
                    torch.zeros((1, label_len + pred_len, 6), dtype=torch.float32).to(
                        device
                    )
                )

            dec_inp = _make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = _slice_pred_target(
                outputs, batch_y, pred_len, features
            )

            # Score only the requested horizon (<= pred_len).
            outputs = outputs[:, :eff_horizon, :]
            batch_y_sliced = batch_y_sliced[:, :eff_horizon, :]

            outputs = outputs.detach().cpu().numpy()
            batch_y_sliced = batch_y_sliced.detach().cpu().numpy()

            if inverse and dataset is not None:
                shape = batch_y_sliced.shape
                outputs = dataset.inverse_transform(
                    outputs.reshape(shape[0] * shape[1], -1)
                ).reshape(shape)
                batch_y_sliced = dataset.inverse_transform(
                    batch_y_sliced.reshape(shape[0] * shape[1], -1)
                ).reshape(shape)

            preds.append(outputs)
            trues.append(batch_y_sliced)

    test_time = time.perf_counter() - start_time
    if not preds:
        raise ValueError("rolling forecast produced no windows")
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = collect_metrics(preds, trues)
    return metrics, test_time
