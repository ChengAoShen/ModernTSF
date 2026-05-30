"""N-BEATS model implementation.

Vendored/adapted from https://github.com/philipperemy/n-beats
(nbeats_pytorch/model.py), MIT License.

N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series
Forecasting (ICLR 2020).

Adapted for ModernTSF: the upstream ``NBeatsNet`` (which carries its own
training loop, ``compile``/``fit``/``predict`` helpers and a
``forward(backcast) -> (backcast, forecast)`` 1-D univariate contract) is
reduced to the pure long-term-forecast architecture and rewrapped to the
ModernTSF ``forward(x_enc, x_mark_enc, ...) -> (B, pred_len, c_out)`` contract.

N-BEATS is channel-independent: each of the ``enc_in`` channels is forecast
independently by the same stack of fully-connected basis-expansion blocks. We
flatten the channel dimension into the batch dimension, run the doubly-residual
stacking, then reshape back. The generic / trend / seasonality blocks and their
basis functions are kept local to this file (no equivalent exists under
``models.module.*``). Trigonometric / polynomial basis matrices are registered
as buffers so they move with ``.to(device)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEASONALITY_BLOCK = "seasonality"
TREND_BLOCK = "trend"
GENERIC_BLOCK = "generic"


def _linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


def _seasonality_basis(thetas_dim, t):
    """Build the [thetas_dim x len(t)] seasonality (Fourier) basis matrix."""
    p = thetas_dim
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])
    S = np.concatenate([s1, s2], axis=0)
    return torch.tensor(S, dtype=torch.float32)


def _trend_basis(thetas_dim, t):
    """Build the [thetas_dim x len(t)] polynomial trend basis matrix."""
    T = np.array([t ** i for i in range(thetas_dim)])
    return torch.tensor(T, dtype=torch.float32)


class Block(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        backcast_length=10,
        forecast_length=5,
        share_thetas=False,
    ):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        if share_thetas:
            self.theta_b_fc = self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class SeasonalityBlock(Block):
    def __init__(
        self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None
    ):
        td = nb_harmonics if nb_harmonics else forecast_length
        super().__init__(
            units, td, backcast_length, forecast_length, share_thetas=True
        )
        b_t = _linear_space(backcast_length, forecast_length, is_forecast=False)
        f_t = _linear_space(backcast_length, forecast_length, is_forecast=True)
        self.register_buffer("backcast_basis", _seasonality_basis(self.thetas_dim, b_t))
        self.register_buffer("forecast_basis", _seasonality_basis(self.thetas_dim, f_t))

    def forward(self, x):
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.backcast_basis)
        forecast = self.theta_f_fc(x).mm(self.forecast_basis)
        return backcast, forecast


class TrendBlock(Block):
    def __init__(
        self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None
    ):
        super().__init__(
            units, thetas_dim, backcast_length, forecast_length, share_thetas=True
        )
        b_t = _linear_space(backcast_length, forecast_length, is_forecast=False)
        f_t = _linear_space(backcast_length, forecast_length, is_forecast=True)
        self.register_buffer("backcast_basis", _trend_basis(self.thetas_dim, b_t))
        self.register_buffer("forecast_basis", _trend_basis(self.thetas_dim, f_t))

    def forward(self, x):
        x = super().forward(x)
        backcast = self.theta_b_fc(x).mm(self.backcast_basis)
        forecast = self.theta_f_fc(x).mm(self.forecast_basis)
        return backcast, forecast


class GenericBlock(Block):
    def __init__(
        self, units, thetas_dim, backcast_length=10, forecast_length=5, nb_harmonics=None
    ):
        super().__init__(units, thetas_dim, backcast_length, forecast_length)
        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = super().forward(x)
        backcast = self.backcast_fc(self.theta_b_fc(x))
        forecast = self.forecast_fc(self.theta_f_fc(x))
        return backcast, forecast


def _select_block(block_type):
    if block_type == SEASONALITY_BLOCK:
        return SeasonalityBlock
    if block_type == TREND_BLOCK:
        return TrendBlock
    return GenericBlock


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        stack_types=(TREND_BLOCK, SEASONALITY_BLOCK, GENERIC_BLOCK),
        nb_blocks_per_stack=3,
        thetas_dim=(4, 8, 8),
        hidden_layer_units=256,
        share_weights_in_stack=False,
        nb_harmonics=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in

        self.stack_types = tuple(stack_types)
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = tuple(thetas_dim)
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics

        assert len(self.thetas_dim) >= len(self.stack_types), (
            "thetas_dim must provide one entry per stack type"
        )

        self.stacks = nn.ModuleList()
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self._create_stack(stack_id))

    def _create_stack(self, stack_id):
        block_cls = _select_block(self.stack_types[stack_id])
        blocks = nn.ModuleList()
        for block_id in range(self.nb_blocks_per_stack):
            if self.share_weights_in_stack and block_id != 0:
                blocks.append(blocks[-1])
            else:
                blocks.append(
                    block_cls(
                        self.hidden_layer_units,
                        self.thetas_dim[stack_id],
                        self.seq_len,
                        self.pred_len,
                        self.nb_harmonics,
                    )
                )
        return blocks

    def _run(self, backcast):
        # backcast: [N, seq_len]
        forecast = backcast.new_zeros((backcast.size(0), self.pred_len))
        for stack in self.stacks:
            for block in stack:
                b, f = block(backcast)
                backcast = backcast - b
                forecast = forecast + f
        return forecast

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, C]; channel-independent.
        B, L, C = x_enc.shape
        x = x_enc.permute(0, 2, 1).reshape(B * C, L)  # [B*C, seq_len]
        forecast = self._run(x)  # [B*C, pred_len]
        forecast = forecast.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred, C]
        if self.features == "MS":
            forecast = forecast[:, :, -1:]
        return forecast
