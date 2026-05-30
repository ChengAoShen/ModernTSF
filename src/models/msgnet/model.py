"""MSGNet model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/MSGNet.py and layers/MSGBlock.py), MIT License.

MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time
Series Forecasting (AAAI 2024).

MSGNet builds an INTERNAL adaptive graph over variates (no external adjacency
input is required): FFT selects the top-k dominant periods, the series is
reshaped per scale, and a learnable mixprop graph convolution mixes information
across channels via an adaptive adjacency matrix derived from learnable node
embeddings. A scale-wise attention block and FFT-amplitude-weighted aggregation
combine the multi-scale representations.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and only the long-term forecast path is kept
(classification / imputation / anomaly-detection branches are dropped). The
shared ``DataEmbedding`` layer under ``models.module.embed`` is reused. The
MSGBlock graph / attention / prediction blocks use upstream-specific
signatures, so they are vendored locally in this file.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.embed import DataEmbedding
from models.module.masking import TriangularCausalMask


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


# --------------------------------------------------------------------------- #
# Vendored MSGBlock layers (upstream layers/MSGBlock.py)
# --------------------------------------------------------------------------- #
class Predict(nn.Module):
    def __init__(self, individual, c_out, seq_len, pred_len, dropout):
        super().__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for _ in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len, pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len, pred_len)
            self.dropout = nn.Dropout(dropout)

    # (B, c_out, seq)
    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:, i, :])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out, dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)
        return out


class _FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        return (V.contiguous(), None)


class _SelfAttention(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention(attention_dropout=0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out, attn


class Attention_Block(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = _SelfAttention(_FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class _nconv(nn.Module):
    def forward(self, x, A):
        x = torch.einsum("ncwl,vw->ncvl", (x, A))
        return x.contiguous()


class _linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

    def forward(self, x):
        return self.mlp(x)


class _mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = _nconv()
        self.mlp = _linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class GraphBlock(nn.Module):
    def __init__(
        self,
        c_out,
        d_model,
        conv_channel,
        skip_channel,
        gcn_depth,
        dropout,
        propalpha,
        seq_len,
        node_dim,
    ):
        super().__init__()
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = _mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    # x in (B, T, d_model)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out, adp))
        out = self.end_conv(out).squeeze(-1)
        out = self.linear(out)
        return self.norm(x + out)


# --------------------------------------------------------------------------- #
# MSGNet core
# --------------------------------------------------------------------------- #
class ScaleGraphBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        top_k,
        d_model,
        d_ff,
        n_heads,
        dropout,
        c_out,
        conv_channel,
        skip_channel,
        gcn_depth,
        propalpha,
        node_dim,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        self.att0 = Attention_Block(
            d_model, d_ff, n_heads=n_heads, dropout=dropout, activation="gelu"
        )
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()
        for _ in range(self.k):
            self.gconv.append(
                GraphBlock(
                    c_out,
                    d_model,
                    conv_channel,
                    skip_channel,
                    gcn_depth,
                    dropout,
                    propalpha,
                    seq_len,
                    node_dim,
                )
            )

    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            # Gconv
            x = self.gconv[i](x)
            # padding
            if self.seq_len % scale != 0:
                length = ((self.seq_len // scale) + 1) * scale
                padding = torch.zeros(
                    [x.shape[0], (length - self.seq_len), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N)

            # for Mul-attention
            out = out.reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)

            out = out[:, : self.seq_len, :]
            res.append(out)

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        label_len=0,
        features="M",
        enc_in=7,
        c_out=None,
        d_model=128,
        d_ff=256,
        e_layers=2,
        n_heads=8,
        top_k=5,
        dropout=0.1,
        conv_channel=32,
        skip_channel=32,
        gcn_depth=2,
        propalpha=0.3,
        node_dim=10,
        individual=False,
        embed="timeF",
        freq="h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        c_out = enc_in if c_out is None else c_out
        self.c_out = c_out

        self.model = nn.ModuleList(
            [
                ScaleGraphBlock(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    top_k=top_k,
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    c_out=c_out,
                    conv_channel=conv_channel,
                    skip_channel=skip_channel,
                    gcn_depth=gcn_depth,
                    propalpha=propalpha,
                    node_dim=node_dim,
                )
                for _ in range(e_layers)
            ]
        )
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.seq2pred = Predict(individual, c_out, seq_len, pred_len, dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)
        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len :, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
