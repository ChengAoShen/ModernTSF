"""Canonical tensor boundary between runners and heterogeneous forecasters."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when DataParallel wraps a forecaster."""
    return model.module if isinstance(model, nn.DataParallel) else model


def make_decoder_input(
    batch_y: torch.Tensor,
    label_len: int,
    pred_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Join the observed decoder label window with a zero future horizon."""
    zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    return torch.cat([batch_y[:, :label_len, :], zeros], dim=1).float().to(device)


def call_forecaster(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_x_mark: torch.Tensor | None,
    decoder_input: torch.Tensor,
    batch_y_mark: torch.Tensor | None,
) -> torch.Tensor:
    """Call the full forecasting signature, falling back to an input-only model."""
    forward = unwrap_model(model).forward
    signature = inspect.signature(forward)
    full_args = (batch_x, batch_x_mark, decoder_input, batch_y_mark)
    try:
        signature.bind(*full_args)
    except TypeError as full_error:
        try:
            signature.bind(batch_x)
        except TypeError:
            raise full_error
        return model(batch_x)
    return model(*full_args)


def slice_prediction_target(
    outputs: torch.Tensor,
    batch_y: torch.Tensor,
    pred_len: int,
    features: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align point or probabilistic output with the forecast target tensor.

    Point outputs have shape ``(B, L, C)``. Quantile and distribution outputs
    add a final parameter axis ``(B, L, C, K)`` that must remain intact while
    the horizon and optional final target channel are selected.
    """
    feature_start = -1 if features == "MS" else 0
    if outputs.dim() == 4:
        outputs = outputs[:, -pred_len:, feature_start:, :]
    else:
        outputs = outputs[:, -pred_len:, feature_start:]
    target = batch_y[:, -pred_len:, feature_start:]
    return outputs, target
