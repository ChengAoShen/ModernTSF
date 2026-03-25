"""PatchTST model implementation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.module.positional_encoding import positional_encoding
from models.module.revin import RevIN
from models.module.tst_transformer import TSTEncoder


class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class PatchTSTModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str] = None,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        activation: str = "gelu",
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        head_dropout: float = 0.0,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = iTSTEncoder(
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            activation=activation,
            norm=norm,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            proj_dropout=proj_dropout,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
        )

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head_type = head_type
        self.individual = individual

        self.head = FlattenHead(
            self.individual,
            self.n_vars,
            self.head_nf,
            target_window,
            head_dropout=head_dropout,
        )

    def forward(self, z):
        if self.revin:
            z = self.revin_layer(z, "norm")

        z = z.permute(0, 2, 1)
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)

        z = self.backbone(z)
        z = self.head(z)
        z = z.permute(0, 2, 1)

        if self.revin:
            z = self.revin_layer(z, "denorm")
        return z


class iTSTEncoder(nn.Module):
    def __init__(
        self,
        patch_num,
        patch_len,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        activation="gelu",
        norm="BatchNorm",
        attn_dropout=0.0,
        res_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len

        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(res_dropout)

        self.encoder = TSTEncoder(
            d_model,
            n_heads,
            n_layers=n_layers,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            activation=activation,
            norm=norm,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            proj_dropout=proj_dropout,
            pre_norm=pre_norm,
        )

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)

        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)
        return z


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str],
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int],
        d_v: Optional[int],
        d_ff: int,
        activation: str,
        norm: str,
        attn_dropout: float,
        res_dropout: float,
        ffn_dropout: float,
        proj_dropout: float,
        head_dropout: float,
        pre_norm: bool,
        pe: str,
        learn_pe: bool,
        head_type: str,
        individual: bool,
        revin: bool,
        affine: bool,
        subtract_last: bool,
    ):
        super().__init__()
        self.model = PatchTSTModel(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            activation=activation,
            norm=norm,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            proj_dropout=proj_dropout,
            head_dropout=head_dropout,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
        )

    def forward(self, x, *args):
        return self.model(x)
