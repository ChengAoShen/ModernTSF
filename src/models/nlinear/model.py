"""NLinear model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


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

        if self.individual:
            self.linear_layer = nn.ModuleList()
            for _ in range(self.channels):
                self.linear_layer.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear_layer = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

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
