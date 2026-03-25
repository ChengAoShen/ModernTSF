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
