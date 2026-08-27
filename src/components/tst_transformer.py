"""Transformer encoder blocks for PatchTST."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if isinstance(activation, str):
        act = activation.lower()
        if act == "relu":
            return nn.ReLU()
        if act == "leakyrelu":
            return nn.LeakyReLU()
        if act == "gelu":
            return nn.GELU()
        if act == "elu":
            return nn.ELU()
        if act == "selu":
            return nn.SELU()
        if act == "prelu":
            return nn.PReLU()
    raise ValueError(
        f"{activation} is not available. Supported: 'relu', 'gelu', 'leakyrelu', 'elu', 'selu', 'prelu', or a callable"
    )


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class TSTEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers=1,
        d_k=None,
        d_v=None,
        d_ff=None,
        activation="gelu",
        norm="BatchNorm",
        attn_dropout=0.0,
        res_dropout=0.0,
        ffn_dropout=0.0,
        proj_dropout=0.0,
        pre_norm=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
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
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        output = src
        for mod in self.layers:
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        activation="gelu",
        norm="BatchNorm",
        attn_dropout=0.0,
        res_dropout=0.0,
        ffn_dropout=0.0,
        proj_dropout=0.0,
        pre_norm=False,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )

        self.dropout_attn = nn.Dropout(res_dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout_ffn = nn.Dropout(res_dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        src2, _ = self.self_attn(
            src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        attn_dropout=0.0,
        proj_dropout=0.0,
        lsa=False,
    ):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout, lsa=lsa
        )

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        output, attn_weights = self.sdp_attn(
            q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )
        output = self.to_out(output)
        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
