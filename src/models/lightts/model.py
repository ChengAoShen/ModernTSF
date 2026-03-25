"""LightTS model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class IEBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        output_dim: int,
        num_node: int,
        c_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node
        if c_dim is None:
            self.c_dim = self.num_node // 2
        else:
            self.c_dim = c_dim
        self._build()

    def _build(self) -> None:
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4),
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class LightTSModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hid_dim: int,
        enc_in: int,
        dropout: float = 0.0,
        chunk_size: int = 40,
        c_dim: int = 40,
    ) -> None:
        super().__init__()
        remainder = seq_len % chunk_size
        lookback = seq_len - remainder
        if lookback <= 0:
            lookback = seq_len

        self.lookback = int(lookback)
        self.lookahead = int(pred_len)
        self.chunk_size = int(chunk_size)
        self.num_chunks = self.lookback // self.chunk_size
        self.hid_dim = int(hid_dim)
        self.num_node = int(enc_in)
        self.c_dim = int(c_dim)
        self.dropout = dropout
        self._build()

    def _build(self) -> None:
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.hid_dim // 4,
            output_dim=self.hid_dim // 4,
            num_node=self.num_chunks,
        )
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)
        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.hid_dim // 4,
            output_dim=self.hid_dim // 4,
            num_node=self.num_chunks,
        )
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)
        self.layer_3 = IEBlock(
            input_dim=self.hid_dim // 2,
            hid_dim=self.hid_dim // 2,
            output_dim=self.lookahead,
            num_node=self.num_node,
            c_dim=self.c_dim,
        )
        self.ar = nn.Linear(self.lookback, self.lookahead)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_node = x.size()
        if seq_len != self.lookback:
            if seq_len >= self.lookback:
                x = x[:, -self.lookback :, :]
            else:
                pad_len = self.lookback - seq_len
                x = torch.nn.functional.pad(x, (0, 0, pad_len, 0))

        highway = self.ar(x.permute(0, 2, 1)).permute(0, 2, 1)

        x1 = x.reshape(batch_size, self.num_chunks, self.chunk_size, num_node)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        x2 = x.reshape(batch_size, self.chunk_size, self.num_chunks, num_node)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(batch_size, num_node, -1)
        x3 = x3.permute(0, 2, 1)
        out = self.layer_3(x3)
        return out + highway


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hid_dim: int,
        dropout: float,
        chunk_size: int,
        c_dim: int,
    ) -> None:
        super().__init__()
        self.model = LightTSModel(
            seq_len=seq_len,
            pred_len=pred_len,
            hid_dim=hid_dim,
            enc_in=enc_in,
            dropout=dropout,
            chunk_size=chunk_size,
            c_dim=c_dim,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.model(x_enc)
