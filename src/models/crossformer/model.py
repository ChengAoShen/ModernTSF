"""Crossformer model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/Crossformer.py), MIT License.

Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate
Time Series Forecasting (ICLR 2023).

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and the only long-term-forecast path is kept
(classification / imputation / anomaly branches dropped). The shared
``PatchEmbedding`` and the attention primitives (``FullAttention`` /
``AttentionLayer``) under ``models.module.*`` are reused. The Crossformer
specific composite blocks (``TwoStageAttentionLayer`` two-stage cross-time /
cross-dimension attention, ``SegMerging``, ``scale_block``, ``Encoder``,
``Decoder``, ``DecoderLayer``) are vendored locally because the upstream
versions take a ``configs`` object and have no matching counterparts in the
shared module library.
"""

from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn
from einops import rearrange, repeat

from models.module.embed import PatchEmbedding
from models.module.self_attention_family import AttentionLayer, FullAttention


class TwoStageAttentionLayer(nn.Module):
    """The Two Stage Attention (TSA) Layer.

    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(
        self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            FullAttention(
                False, factor, attention_dropout=dropout, output_attention=False
            ),
            d_model,
            n_heads,
        )
        self.dim_sender = AttentionLayer(
            FullAttention(
                False, factor, attention_dropout=dropout, output_attention=False
            ),
            d_model,
            n_heads,
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(
                False, factor, attention_dropout=dropout, output_attention=False
            ),
            d_model,
            n_heads,
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(
            x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model"
        )
        time_enc, _ = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to
        # aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(
            dim_in,
            "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model",
            b=batch,
        )
        batch_router = repeat(
            self.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )
        dim_buffer, _ = self.dim_sender(
            batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None
        )
        dim_receive, _ = self.dim_receiver(
            dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None
        )
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(
            dim_enc,
            "(b seg_num) ts_d d_model -> b ts_d seg_num d_model",
            b=batch,
        )

        return final_out


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i :: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    def __init__(
        self,
        win_size,
        d_model,
        n_heads,
        d_ff,
        depth,
        dropout,
        seg_num=10,
        factor=10,
    ):
        super().__init__()

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for _ in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(
                    seg_num, factor, d_model, n_heads, d_ff, dropout
                )
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x, None


class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super().__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x, _ = block(x)
            encode_x.append(x)

        return encode_x, None


class DecoderLayer(nn.Module):
    def __init__(
        self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(
            x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model"
        )

        cross = rearrange(
            cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model"
        )
        tmp, _ = self.cross_attention(x, cross, cross, None, None, None)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(
            dec_output,
            "(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model",
            b=batch,
        )
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(
            layer_predict, "b out_d seg_num seg_len -> b (out_d seg_num) seg_len"
        )

        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(
            final_predict,
            "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d",
            out_d=ts_d,
        )

        return final_predict


class Model(nn.Module):
    """Crossformer (long-term forecast)."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        seg_len=12,
        win_size=2,
        factor=10,
        dropout=0.1,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.seg_len = seg_len
        self.win_size = win_size

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * seq_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * pred_len / seg_len) * seg_len
        self.in_seg_num = self.pad_in_len // seg_len
        self.out_seg_num = ceil(self.in_seg_num / (win_size ** (e_layers - 1)))
        self.head_nf = d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            d_model, seg_len, seg_len, self.pad_in_len - seq_len, 0
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, self.in_seg_num, d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    1 if l == 0 else win_size,
                    d_model,
                    n_heads,
                    d_ff,
                    1,
                    dropout,
                    self.in_seg_num
                    if l == 0
                    else ceil(self.in_seg_num / win_size**l),
                    factor,
                )
                for l in range(e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, enc_in, (self.pad_out_len // seg_len), d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        (self.pad_out_len // seg_len),
                        factor,
                        d_model,
                        n_heads,
                        d_ff,
                        dropout,
                    ),
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
                    seg_len,
                    d_model,
                    d_ff,
                    dropout=dropout,
                )
                for _ in range(e_layers + 1)
            ]
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, _ = self.encoder(x_enc)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_enc.shape[0],
        )
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
