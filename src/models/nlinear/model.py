"""NLinear model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from components.channel_wise_linear import ChannelWiseLinear


class NLinearModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c_in
        self.individual = individual

        self.projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        x = x.permute(0, 2, 1)

        output = self.projection(x)

        output = output.permute(0, 2, 1)
        output = output + seq_last
        return output


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
    ):
        super().__init__()
        self.model = NLinearModel(
            c_in=c_in,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
        )

    def forward(self, x, *args):
        return self.model(x)
