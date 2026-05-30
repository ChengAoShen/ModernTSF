"""SRSNet model implementation.

Vendored/adapted from https://github.com/decisionintelligence/SRSNet
(ts_benchmark/baselines/srsnet/models/srsnet_model.py and
ts_benchmark/baselines/srsnet/layers/SRS.py), MIT License
(Copyright (c) 2024 Huawei Technologies Co., Ltd).

SRSNet: Enhancing Time Series Forecasting through Selective Representation
Spaces: A Patch Perspective (NeurIPS 2025 Spotlight). The Selective
Representation Space (SRS) module builds two patch views of the input -- the
plain (original) striding view and a learnable "reconstruction" view that uses
Selective Patching (picks informative dense-stride patches) and Dynamic
Reassembly (re-orders them by a learned score) -- then adaptively fuses the two
view embeddings. A flatten + MLP head produces the forecast.

Adapted for ModernTSF: the upstream ``config``-object constructor is replaced
with plain keyword arguments, and the shared ``RevIN`` layer under
``models.module.revin`` is reused. The ``SRS`` block and ``FlattenHead`` are
SRSNet-specific and kept local to this file (the upstream local
``PositionalEmbedding`` is identical to the shared
``models.module.embed.PositionalEmbedding`` and is imported from there).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from models.module.embed import PositionalEmbedding
from models.module.revin import RevIN


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode="linear"):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        if mode == "linear":
            self.head = nn.Linear(nf, target_window)
        else:
            self.head = nn.Sequential(
                nn.Linear(nf, nf // 2), nn.SiLU(), nn.Linear(nf // 2, target_window)
            )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)
        return x


class SRS(nn.Module):
    """Selective Representation Space: fuses an original-stride patch view with a
    learnable selected-and-shuffled patch view."""

    def __init__(
        self, d_model, patch_len, stride, seq_len, dropout, hidden_size, alpha=2.0, pos=True
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len

        self.patch_num = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.padding = (
            self.patch_len + (self.patch_num - 1) * self.stride - self.seq_len
        )
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))
        self.scorer_select = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.patch_num),
        )

        self.scorer_shuffle = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        if pos:
            self.position_embedding = PositionalEmbedding(d_model)

        self.pos = pos

        self.dropout = nn.Dropout(dropout)

        # Adaptive weight between Original View and Reconstruction View
        self.alpha = nn.Parameter(torch.ones(self.patch_num, d_model) * alpha)

    def _origin_view(self, x):
        # [batch_size, n_vars, patch_num, patch_size]
        x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [batch_size * n_vars, patch_num, patch_size]
        origin_patches = rearrange(x_origin, "b c n p -> (b c) n p")
        return origin_patches

    def _rec_view(self, x):
        # [batch_size, n_vars, seq_len - patch_size + 1, patch_size]
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)
        # [batch_size, n_vars, patch_num, patch_size]
        selected_patches = self._select(x_rec)
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = self._shuffle(selected_patches)
        # [batch_size * n_vars, patch_num, patch_size]
        rec_patches = rearrange(shuffled_patches, "b c n p -> (b c) n p")
        return rec_patches

    def _select(self, x_rec):
        # [batch_size, n_vars, seq_len - patch_size + 1, select_num]
        scores = self.scorer_select(x_rec)
        # [batch_size, n_vars, 1, select_num]
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        # [batch_size, n_vars, 1, select_num]
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, select_num, patch_size]
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        # [batch_size, n_vars, select_num, patch_size]
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)

        max_scores[non_zero_mask] *= inv
        # [batch_size, n_vars, select_num, patch_size]
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches

        return selected_patches

    def _shuffle(self, selected_patches):
        # [batch_size, n_vars, patch_num, 1]
        shuffle_scores = self.scorer_shuffle(selected_patches)
        # [batch_size, n_vars, patch_num, 1]
        shuffle_indices = torch.argsort(input=shuffle_scores, dim=-2, descending=True)
        # [batch_size, n_vars, patch_num, 1]
        shuffled_scores = torch.gather(
            input=shuffle_scores, index=shuffle_indices, dim=-2
        )
        non_zero_mask = shuffled_scores != 0
        inv = (1 / shuffled_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, patch_num, patch_size]
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = torch.gather(
            input=selected_patches, index=shuffle_patch_indices, dim=-2
        )
        shuffled_scores[non_zero_mask] *= inv
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = shuffled_scores * shuffled_patches

        return shuffled_patches

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # padding for the original stride
        x = self.padding_patch_layer(x)

        # [batch_size * n_vars, patch_num, patch_size]
        rec_repr_space = self._rec_view(x)
        # [batch_size * n_vars, patch_num, patch_size]
        original_repr_space = self._origin_view(x)
        # The adaptive weight between the two views
        weight = torch.sigmoid(self.alpha)
        # [batch_size * n_vars, patch_num, d_model]
        embedding = weight * self.value_embedding_org(original_repr_space) + (
            1 - weight
        ) * self.value_embedding_rec(rec_repr_space)

        if self.pos:
            position_embedding = self.position_embedding(original_repr_space)
            embedding = embedding + position_embedding

        return self.dropout(embedding), n_vars


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=512,
        patch_len=24,
        stride=24,
        hidden_size=128,
        dropout=0.2,
        head_dropout=0.1,
        alpha=2.0,
        pos=True,
        head_mode="linear",
        affine=True,
        subtract_last=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # selective representation space
        self.patch_embedding = SRS(
            d_model,
            self.patch_len,
            self.stride,
            self.seq_len,
            dropout,
            hidden_size,
            alpha,
            pos,
        )

        # Prediction Head
        self.head_nf = d_model * (
            math.ceil((seq_len - self.patch_len) / self.stride) + 1
        )
        self.head = FlattenHead(
            enc_in,
            self.head_nf,
            pred_len,
            head_dropout=head_dropout,
            mode=head_mode,
        )

        self.revin = RevIN(
            num_features=enc_in, affine=affine, subtract_last=subtract_last
        )

    def forecast(self, x_enc):
        x_enc = self.revin(x_enc, "norm")
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization
        dec_out = self.revin(dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
