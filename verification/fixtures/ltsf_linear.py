"""Thin LTSF-Linear fixtures pinned to the recorded upstream revision.

Source: https://github.com/cure-lab/LTSF-Linear
Revision: 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6
License: Apache-2.0, Copyright 2022 DLinear Authors

The classes below retain the defining forward paths and state-dict names from
``models/{Linear,NLinear,DLinear}.py``. Imports, comments, and configuration
plumbing that do not affect those paths are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


SOURCE_URL = "https://github.com/cure-lab/LTSF-Linear"
SOURCE_REVISION = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
SOURCE_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class Config:
    seq_len: int
    pred_len: int
    enc_in: int
    individual: bool = False


class Linear(nn.Module):
    """Upstream ``models/Linear.py::Model`` forward and state layout."""

    def __init__(self, configs: Config):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList(
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.channels)
            )
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)],
                dtype=x.dtype,
                device=x.device,
            )
            for index in range(self.channels):
                output[:, :, index] = self.Linear[index](x[:, :, index])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class NLinear(Linear):
    """Upstream ``models/NLinear.py::Model`` forward and state layout."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)],
                dtype=x.dtype,
                device=x.device,
            )
            for index in range(self.channels):
                output[:, :, index] = self.Linear[index](x[:, :, index])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + seq_last


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class DLinear(nn.Module):
    """Upstream ``models/DLinear.py::Model`` forward and state layout."""

    def __init__(self, configs: Config):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.decompsition = SeriesDecomp(25)
        self.individual = configs.individual
        self.channels = configs.enc_in
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.channels)
            )
            self.Linear_Trend = nn.ModuleList(
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.channels)
            )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for index in range(self.channels):
                seasonal_output[:, index, :] = self.Linear_Seasonal[index](
                    seasonal_init[:, index, :]
                )
                trend_output[:, index, :] = self.Linear_Trend[index](
                    trend_init[:, index, :]
                )
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)


__all__ = [
    "Config",
    "DLinear",
    "Linear",
    "NLinear",
    "SOURCE_LICENSE",
    "SOURCE_REVISION",
    "SOURCE_URL",
]
