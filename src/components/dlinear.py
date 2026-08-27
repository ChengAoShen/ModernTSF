"""Series decomposition and linear forecasting backbone used by DLinear methods."""

from __future__ import annotations

import torch
import torch.nn as nn

from components.series_decomposition import MovingAvg, SeriesDecomp


class DLinearBackbone(nn.Module):
    """DLinear sequence-to-sequence forecasting backbone.

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
