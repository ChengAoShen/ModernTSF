"""Fredformer model implementation.

Vendored/adapted from https://github.com/chenzRG/Fredformer
(models/Fredformer.py, layers/Fredformer_backbone.py, layers/cross_Transformer.py),
no explicit license file (research code released alongside the KDD 2024 paper
"Fredformer: Frequency Debiased Transformer for Time Series Forecasting").

Fredformer applies the FFT to the look-back window, patches the real and
imaginary spectra, runs a channel-wise ("Nystrom-free") frequency transformer
over the concatenated real/imaginary patches, and reconstructs the forecast via
an inverse FFT.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, the shared ``models.module.revin.RevIN`` layer is
reused, and only the standard (non-Nystrom, non-ablation) long-term forecast
path is kept. The frequency-domain cross-channel transformer blocks
(``Trans_C`` / ``c_Transformer`` / ``c_Attention``) are Fredformer-specific and
are vendored locally in this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.module.revin import RevIN


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) / self.d_k

        attn = self.attend(dots)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out), attn


class c_Transformer(nn.Module):
    """Channel-wise frequency transformer (registers the attention/FFN blocks)."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            c_Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        attn = None
        for attn_block, ff in self.layers:
            x_n, attn = attn_block(x)
            x = x_n + x
            x = ff(x) + x
        return x, attn


class Trans_C(nn.Module):
    def __init__(
        self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model
    ):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim), nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x, _attn = self.transformer(x)
        x = self.dropout(x)
        x = self.mlp_head(x)
        return x


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x patch_num x d_model]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x


class Fredformer_backbone(nn.Module):
    def __init__(
        self,
        c_in,
        context_window,
        target_window,
        patch_len,
        stride,
        d_model,
        cf_dim,
        cf_depth,
        cf_heads,
        cf_mlp,
        cf_head_dim,
        cf_drop,
        head_dropout=0,
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
    ):
        super().__init__()
        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len) / stride + 1)
        self.norm = nn.LayerNorm(patch_len)

        # Backbone (standard / non-Nystrom path)
        self.fre_transformer = Trans_C(
            dim=cf_dim,
            depth=cf_depth,
            heads=cf_heads,
            mlp_dim=cf_mlp,
            dim_head=cf_head_dim,
            dropout=cf_drop,
            patch_dim=patch_len * 2,
            horizon=self.horizon * 2,
            d_model=d_model * 2,
        )

        # Head
        self.head_nf_f = d_model * 2 * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(
            individual, self.n_vars, self.head_nf_f, target_window,
            head_dropout=head_dropout,
        )
        self.head_f2 = Flatten_Head(
            individual, self.n_vars, self.head_nf_f, target_window,
            head_dropout=head_dropout,
        )

        self.ircom = nn.Linear(self.targetwindow * 2, self.targetwindow)

        # break up real & imaginary projections
        self.get_r = nn.Linear(d_model * 2, d_model * 2)
        self.get_i = nn.Linear(d_model * 2, d_model * 2)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # for channel-wise
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))

        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # [bs, nvars, patch_num, d_model*2]
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)  # [bs, nvars, target_window]
        z2 = self.head_f2(z2)

        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        d_model=16,
        e_layers=2,
        n_heads=8,
        d_ff=128,
        dropout=0.1,
        patch_len=16,
        stride=8,
        revin=True,
        affine=True,
        subtract_last=False,
        individual=False,
        head_dropout=0.0,
        cf_dim=48,
        cf_depth=2,
        cf_heads=6,
        cf_mlp=128,
        cf_head_dim=32,
        cf_drop=0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features

        self.model = Fredformer_backbone(
            c_in=enc_in,
            context_window=seq_len,
            target_window=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            cf_dim=cf_dim,
            cf_depth=cf_depth,
            cf_heads=cf_heads,
            cf_mlp=cf_mlp,
            cf_head_dim=cf_head_dim,
            cf_drop=cf_drop,
            head_dropout=head_dropout,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, C]
        x = x_enc.permute(0, 2, 1)  # [B, C, seq_len]
        x = self.model(x)  # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        return x[:, -self.pred_len :, :]
