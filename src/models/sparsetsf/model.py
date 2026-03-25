"""SparseTSF model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseTSFModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period: int,
        d_model: int,
        model_type: str,
    ):
        super().__init__()
        self.seq_len = seq_len // period * period
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period = period
        self.d_model = d_model
        self.model_type = model_type

        self.seg_num_x = self.seq_len // self.period
        self.seg_num_y = self.pred_len // self.period

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period // 2),
            stride=1,
            padding=self.period // 2,
            padding_mode="zeros",
            bias=False,
        )

        if self.model_type == "linear":
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y),
            )
        else:
            raise ValueError("model_type must be 'linear' or 'mlp'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : self.seq_len, :]
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        x = (
            self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
                -1, self.enc_in, self.seq_len
            )
            + x
        )

        x = x.reshape(-1, self.seg_num_x, self.period).permute(0, 2, 1)

        if self.model_type == "linear":
            y = self.linear(x)
        else:
            y = self.mlp(x)

        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_mean
        return y


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period: int,
        d_model: int,
        model_type: str,
    ):
        super().__init__()
        self.model = SparseTSFModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            period=period,
            d_model=d_model,
            model_type=model_type,
        )

    def forward(self, x, *args):
        return self.model(x)
