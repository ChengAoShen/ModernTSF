"""TexFilter model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.revin import RevIN


class TexFilterModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        embed_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.band_width = 96
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.token = nn.Conv1d(
            in_channels=self.seq_len, out_channels=self.embed_size, kernel_size=(1,)
        )

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size),
        )

        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def token_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.token(x)

    def texfilter(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        o1_real = torch.zeros([b, n // 2 + 1, self.embed_size], device=x.device)
        o1_imag = torch.zeros([b, n // 2 + 1, self.embed_size], device=x.device)

        o2_real = torch.zeros([b, n // 2 + 1, self.embed_size], device=x.device)
        o2_imag = torch.zeros([b, n // 2 + 1, self.embed_size], device=x.device)

        o1_real = F.relu(
            torch.einsum("bid,d->bid", x.real, self.w[0])
            - torch.einsum("bid,d->bid", x.imag, self.w[1])
            + self.rb1
        )

        o1_imag = F.relu(
            torch.einsum("bid,d->bid", x.imag, self.w[0])
            + torch.einsum("bid,d->bid", x.real, self.w[1])
            + self.ib1
        )

        o2_real = (
            torch.einsum("bid,d->bid", o1_real, self.w1[0])
            - torch.einsum("bid,d->bid", o1_imag, self.w1[1])
            + self.rb2
        )

        o2_imag = (
            torch.einsum("bid,d->bid", o1_imag, self.w1[0])
            + torch.einsum("bid,d->bid", o1_real, self.w1[1])
            + self.ib2
        )

        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n = x.shape
        z = self.revin_layer(x, "norm")

        x = z.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.layernorm(x)
        x = torch.fft.rfft(x, dim=1, norm="ortho")

        weight = self.texfilter(x)
        x = x * weight
        x = torch.fft.irfft(x, n=n, dim=1, norm="ortho")
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)

        z = self.revin_layer(x, "denorm")
        return z


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        embed_size: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.model = TexFilterModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, x, *args):
        return self.model(x)
