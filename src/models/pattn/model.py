"""PAttn model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/PAttn.py), MIT License.

PAttn: "Are Language Models Actually Useful for Time Series Forecasting?"
(NeurIPS 2024, https://arxiv.org/abs/2406.16964). A deliberately simple
patch-based baseline: pad + unfold the input into overlapping patches, embed
each patch with a linear layer, run a single Transformer self-attention
encoder block over the patch tokens (per channel), then flatten and linearly
project to the forecast horizon.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and the shared attention layers under
``models.module.self_attention_family`` (``FullAttention``, ``AttentionLayer``)
are reused. The composite ``Encoder`` / ``EncoderLayer`` (upstream
``layers/Transformer_EncDec.py``) are vendored locally below — only the
encoder-only, conv-free path PAttn needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.module.self_attention_family import AttentionLayer, FullAttention


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=128,
        n_heads=8,
        d_ff=256,
        patch_len=16,
        stride=8,
        dropout=0.1,
        factor=3,
        activation="gelu",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.patch_size = patch_len
        self.stride = stride
        self.d_model = d_model

        self.patch_num = (seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(self.patch_size, d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.out_layer = nn.Linear(d_model * self.patch_num, pred_len)

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        B, _, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        enc_out = self.in_layer(x_enc)
        enc_out = rearrange(enc_out, "b c m l -> (b c) m l")
        dec_out, _ = self.encoder(enc_out)
        dec_out = rearrange(dec_out, "(b c) m l -> b c (m l)", b=B, c=C)
        dec_out = self.out_layer(dec_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
