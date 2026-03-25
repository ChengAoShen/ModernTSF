"""DLinear model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average block for series smoothing.

    Parameters
    ----------
    kernel_size : int, optional
        Window size for averaging.
    stride : int, optional
        Stride for average pooling.
    """

    def __init__(self, kernel_size: int = 24, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply moving average along the temporal axis.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C).

        Returns
        -------
        torch.Tensor
            Smoothed tensor of shape (B, L, C).
        """
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Decompose a series into residual and trend components."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        """Return residual and moving average components.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Residual and trend components.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """DLinear model for sequence-to-sequence forecasting.

    Parameters
    ----------
    c_in : int
        Number of input channels.
    seq_len : int
        Input sequence length.
    pred_len : int
        Prediction horizon length.
    kernel_size : int, optional
        Kernel size for the moving average decomposition.
    individual : bool, optional
        Whether to use per-channel linear layers.
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = c_in

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.channels):
                self.linear_seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.linear_trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a prediction sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, pred_len, C).
        """
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                (seasonal_init.size(0), seasonal_init.size(1), self.pred_len),
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                (trend_init.size(0), trend_init.size(1), self.pred_len),
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        out = seasonal_output + trend_output
        return out.permute(0, 2, 1)
