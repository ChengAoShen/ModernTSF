"""Independent differentiable fixed-gain alpha-beta forecasting filter."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class Model(nn.Module):
    """Learn per-channel fixed gains for a constant-velocity state filter."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, initial_alpha: float = 0.5, initial_beta: float = 0.25) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in) < 1:
            raise ValueError("dimensions must be positive")
        if not 0 < initial_alpha < 1 or not 0 < initial_beta < 1:
            raise ValueError("initial gains must lie strictly between zero and one")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.alpha_logits = nn.Parameter(torch.full((enc_in,), _logit(initial_alpha)))
        self.beta_logits = nn.Parameter(torch.full((enc_in,), _logit(initial_beta)))
        self.aux_loss: None = None

    def forward(self, x: torch.Tensor, *args: object) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}")
        alpha, beta = self.alpha_logits.sigmoid(), self.beta_logits.sigmoid()
        level, velocity = x[:, 0], torch.zeros_like(x[:, 0])
        for index in range(1, self.seq_len):
            predicted_level = level + velocity
            innovation = x[:, index] - predicted_level
            level = predicted_level + alpha * innovation
            velocity = velocity + beta * innovation
        horizon = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype)
        return level[:, None, :] + horizon[None, :, None] * velocity[:, None, :]
