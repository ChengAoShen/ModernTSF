"""Reformer model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/Reformer.py), MIT License.

Reformer: The Efficient Transformer (ICLR 2020,
https://openreview.net/forum?id=rkgNKkHtvB). An encoder-only forecaster that
replaces full self-attention with LSH (locality-sensitive hashing) attention
for O(L log L) complexity.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and only the long-term forecast path is kept
(classification / imputation / anomaly-detection branches are dropped). The
shared ``Encoder`` / ``EncoderLayer`` (``models.module.transformer_encdec``)
and ``DataEmbedding`` (``models.module.embed``) leaf layers are reused.

The ``ReformerLayer`` (LSH self-attention) is vendored locally as a
self-contained implementation so the model has no external runtime
dependency (the shared ``models.module.self_attention_family.ReformerLayer``
requires the optional ``reformer_pytorch`` package).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.module.embed import DataEmbedding
from models.module.transformer_encdec import Encoder, EncoderLayer


class LSHSelfAttention(nn.Module):
    """Minimal locality-sensitive-hashing self-attention.

    Self-contained re-implementation of the LSH attention used by Reformer,
    avoiding the optional ``reformer_pytorch`` dependency. Queries and keys
    share a projection (as in the original Reformer), sequences are bucketed
    by random-projection hashing, and attention is computed within buckets.
    """

    def __init__(self, dim, heads=8, bucket_size=4, n_hashes=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes

        self.to_qk = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _hash_vectors(self, vecs, n_buckets):
        # vecs: [B*H, L, head_dim] -> bucket id per position for each hash round
        b, l, d = vecs.shape
        rotations = torch.randn(
            self.n_hashes, d, n_buckets // 2, device=vecs.device, dtype=vecs.dtype
        )
        # [n_hashes, B*H, L, n_buckets//2]
        rotated = torch.einsum("bld,hde->hble", vecs, rotations)
        rotated = torch.cat([rotated, -rotated], dim=-1)  # [.., n_buckets]
        buckets = torch.argmax(rotated, dim=-1)  # [n_hashes, B*H, L]
        return buckets

    def forward(self, x):
        # x: [B, L, dim]
        b, l, _ = x.shape
        h, hd = self.heads, self.head_dim

        qk = self.to_qk(x).view(b, l, h, hd).transpose(1, 2).reshape(b * h, l, hd)
        v = self.to_v(x).view(b, l, h, hd).transpose(1, 2).reshape(b * h, l, hd)

        # normalise qk for stable hashing / dot products
        qk = qk / (qk.norm(dim=-1, keepdim=True) + 1e-6)

        n_buckets = max(2, (l // self.bucket_size) // 1 * 2)
        n_buckets = n_buckets if n_buckets % 2 == 0 else n_buckets + 1

        buckets = self._hash_vectors(qk, n_buckets)  # [n_hashes, B*H, L]

        out_accum = torch.zeros_like(v)
        for r in range(self.n_hashes):
            bk = buckets[r]  # [B*H, L]
            # mask[i, j] = True where same bucket (attend), per batch element
            same = bk.unsqueeze(-1) == bk.unsqueeze(-2)  # [B*H, L, L]
            scores = torch.matmul(qk, qk.transpose(-1, -2)) / math.sqrt(hd)
            scores = scores.masked_fill(~same, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)
            out_accum = out_accum + torch.matmul(attn, v)

        out = out_accum / self.n_hashes  # average over hash rounds
        out = out.view(b, h, l, hd).transpose(1, 2).reshape(b, l, self.dim)
        return self.to_out(out)


class ReformerLayer(nn.Module):
    """Encoder attention wrapper exposing the standard (q, k, v) interface.

    Pads the sequence length to a multiple of ``2 * bucket_size`` (as required
    by LSH bucketing), applies LSH self-attention, then crops back.
    """

    def __init__(
        self,
        attention,  # unused; kept for upstream signature compatibility
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        causal=False,
        bucket_size=4,
        n_hashes=4,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
        )

    def fit_length(self, queries):
        batch_size, length, channels = queries.shape
        if length % (self.bucket_size * 2) == 0:
            return queries
        fill_len = (self.bucket_size * 2) - (length % (self.bucket_size * 2))
        return torch.cat(
            [
                queries,
                torch.zeros([batch_size, fill_len, channels], device=queries.device),
            ],
            dim=1,
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch_size, length, channels = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :length, :]
        return queries, None


class Model(nn.Module):
    """Reformer with O(L log L) LSH attention (encoder-only forecaster)."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        c_out=None,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        activation="gelu",
        embed="timeF",
        freq="h",
        bucket_size=4,
        n_hashes=4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features
        c_out = c_out if c_out is not None else enc_in

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(
                        None,
                        d_model,
                        n_heads,
                        bucket_size=bucket_size,
                        n_hashes=n_hashes,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # add placeholder: append zero-future region to the encoder input
        zeros = torch.zeros(
            x_enc.shape[0], self.pred_len, x_enc.shape[2], device=x_enc.device
        )
        if x_dec is not None:
            zeros = x_dec[:, -self.pred_len :, :]
        x_enc = torch.cat([x_enc, zeros], dim=1)
        if x_mark_enc is not None and x_mark_dec is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len :, :]], dim=1
            )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out  # [B, L, D]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
