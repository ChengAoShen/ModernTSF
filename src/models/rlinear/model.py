"""RLinear model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from components.channel_wise_linear import ChannelWiseLinear
from components.revin import RevIN


class RLinearModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
        affine: bool = False,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = c_in

        self.projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)

        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channel]
        x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)

        output = self.projection(x)

        output = output.permute(0, 2, 1)
        output = self.revin_layer(output, "denorm")
        return output


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
        affine: bool = False,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.model = RLinearModel(
            c_in=c_in,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
            affine=affine,
            subtract_last=subtract_last,
        )

    def forward(self, x, *args):
        return self.model(x)
