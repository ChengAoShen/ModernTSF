"""RLinear model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN


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

        if self.individual:
            self.linear_layer = nn.ModuleList()
            for _ in range(self.channels):
                self.linear_layer.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear_layer = nn.Linear(self.seq_len, self.pred_len)

        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channel]
        x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)

        if self.individual:
            output = torch.zeros(
                [x.size(0), x.size(1), self.pred_len],
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.channels):
                output[:, i, :] = self.linear_layer[i](x[:, i, :])
        else:
            output = self.linear_layer(x)

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
