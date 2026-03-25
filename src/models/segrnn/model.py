"""SegRNN model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegRNNModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        dropout: float,
        seg_len: int,
        task_name: str = "long_term_forecast",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout
        self.task_name = task_name
        if self.task_name in ("classification", "anomaly_detection", "imputation"):
            self.pred_len = self.seq_len
        else:
            self.pred_len = pred_len

        self.seg_len = seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        self.value_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)
        total_len = self.seg_num_x * self.seg_len
        seq_len = x.shape[-1]
        if seq_len > total_len:
            x = x[:, :, -total_len:]
        elif seq_len < total_len:
            pad_len = total_len - seq_len
            x = F.pad(x, (pad_len, 0))

        x = self.value_embedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        _, hn = self.rnn(x)

        pos_emb = (
            torch.cat(
                [
                    self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
                ],
                dim=-1,
            )
            .view(-1, 1, self.d_model)
            .repeat(batch_size, 1, 1)
        )

        _, hy = self.rnn(
            pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        )

        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_enc)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        dropout: float,
        seg_len: int,
    ) -> None:
        super().__init__()
        self.model = SegRNNModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            dropout=dropout,
            seg_len=seg_len,
            task_name="long_term_forecast",
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
