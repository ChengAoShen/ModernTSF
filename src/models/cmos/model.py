"""CMoS model implementation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CMoSModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        seg_size: int,
        num_map: int,
        kernel_size: int,
        conv_stride: int,
        topk: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seg_size = seg_size
        self.num_map = num_map
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.c_in = c_in
        self.topk = topk

        self.mappings = nn.ModuleList(
            [
                nn.Linear(seq_len // self.seg_size, pred_len // self.seg_size)
                for _ in range(self.num_map)
            ]
        )

        self.conv_dim = (seq_len - self.kernel_size) // self.conv_stride + 1
        self.ds_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=self.kernel_size,
                    stride=self.conv_stride,
                )
                for _ in range(self.c_in)
            ]
        )

        self.gates = nn.Linear(self.conv_dim, self.num_map)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channel]
        x = x.transpose(-2, -1)

        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-10)
        x = x / stdev

        conv_outs = [
            self.ds_convs[i](x[:, i, :].unsqueeze(1)) for i in range(self.c_in)
        ]
        conv_out = torch.cat(conv_outs, dim=1)

        gates_out = self.gates(conv_out.squeeze(1))
        gates_out = F.softmax(gates_out, dim=-1)

        if self.topk > 0:
            gates_out, topk_indice = torch.topk(gates_out, self.topk, dim=-1)
            gates_out = F.softmax(gates_out, dim=-1)

        bs, c, _ = x.shape
        x_ = x.reshape(bs, c, -1, self.seg_size).transpose(2, 3)

        if self.topk > 0:
            new_gates_out = torch.zeros(bs, c, self.num_map, device=x.device)
            new_gates_out.scatter_(-1, topk_indice, gates_out)
            gates_out = new_gates_out

        x_out = [
            self.mappings[i](x_).transpose(2, 3).flatten(start_dim=2)
            for i in range(self.num_map)
        ]
        x_out = torch.stack(x_out, dim=2)
        x = torch.einsum("bcns,bcn->bcs", x_out, gates_out)

        x = x * stdev
        x = x + means
        return x.transpose(-2, -1)


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        seg_size: int,
        num_map: int,
        kernel_size: int,
        conv_stride: int,
        topk: int,
        dropout: float,
    ):
        super().__init__()
        self.model = CMoSModel(
            seq_len=seq_len,
            pred_len=pred_len,
            c_in=c_in,
            seg_size=seg_size,
            num_map=num_map,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            topk=topk,
            dropout=dropout,
        )

    def forward(self, x, *args):
        return self.model(x)
