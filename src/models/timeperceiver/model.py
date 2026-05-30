"""TimePerceiver model implementation.

Vendored/adapted from https://github.com/efficient-learning-lab/TimePerceiver
(models/TimePerceiver.py), MIT License.

TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series
Forecasting (NeurIPS 2025). A Perceiver-style architecture that patches the
input, compresses it through a small set of learnable latent vectors via
iterative cross/self attention, then decodes future patches with a query
cross-attention.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and only the long-term forecasting path is kept
(the ``indices`` masking branch used for generalized/variable-horizon training
is dropped — past patches use ``[:past_patch_num]`` and the query uses
``[past_patch_num:]``). The ``CrossAttention`` / ``AttentionBlock`` blocks are
TimePerceiver-specific and kept local to this file. RevIN-style instance
normalisation is applied inline, matching upstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, num_heads, query_dim, key_value_dim, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.key_value_dim = key_value_dim
        self.query_dim = query_dim
        self.dropout = nn.Dropout(dropout_rate)

        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be a multiple of num_heads.")

        self.head_size = query_dim // num_heads

        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_value_dim, query_dim)
        self.value = nn.Linear(key_value_dim, query_dim)
        self.out = nn.Sequential(nn.Linear(query_dim, query_dim), self.dropout)

    def forward(self, query_input, key_value_input):
        batch_size = query_input.shape[0]
        query_len = query_input.shape[1]
        key_value_len = key_value_input.shape[1]

        query = self.query(query_input)
        key = self.key(key_value_input)
        value = self.value(key_value_input)

        query = query.view(
            batch_size, query_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        key = key.view(
            batch_size, key_value_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        value = value.view(
            batch_size, key_value_len, self.num_heads, self.head_size
        ).transpose(1, 2)

        score_matrix = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        )
        attention_matrix = torch.softmax(score_matrix, dim=-1)
        attention_matrix = self.dropout(attention_matrix)

        result_matrix = torch.matmul(attention_matrix, value)
        result_matrix = (
            result_matrix.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.query_dim)
        )

        return self.out(result_matrix)


class AttentionBlock(nn.Module):
    def __init__(self, num_heads, query_dim, key_value_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.key_value_norm = nn.LayerNorm(key_value_dim)

        self.attention = CrossAttention(num_heads, query_dim, key_value_dim, dropout_rate)
        self.layer_norm2 = nn.LayerNorm(query_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, query_dim),
        )

    def forward(self, query_input, key_value_input):
        query_input = query_input + self.dropout(
            self.attention(
                self.query_norm(query_input), self.key_value_norm(key_value_input)
            )
        )
        query_input = query_input + self.dropout(self.mlp(self.layer_norm2(query_input)))
        return query_input


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        d_model=32,
        n_heads=2,
        d_ff=256,
        patch_len=16,
        dropout=0.1,
        num_latents=8,
        latent_dim=128,
        latent_d_ff=256,
        num_latent_blocks=1,
        use_latent=True,
        query_share=True,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})."
            )
        if pred_len % patch_len != 0:
            raise ValueError(
                f"pred_len ({pred_len}) must be divisible by patch_len ({patch_len})."
            )

        self.patch_size = patch_len
        self.past_patch_num = seq_len // patch_len
        self.future_patch_num = pred_len // patch_len
        self.embed_dim = d_model
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.query_share = query_share

        # Latent
        self.use_latent = use_latent
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_latent_blocks = num_latent_blocks
        self.latent_array = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # Positional embeddings (time + channel directions)
        self.patch_positional_embedding = nn.Parameter(
            torch.randn(
                1, 1, self.past_patch_num + self.future_patch_num, self.embed_dim
            )
        )
        self.channel_positional_embedding = nn.Parameter(
            torch.randn(1, enc_in, 1, self.embed_dim)
        )

        self.patch_embedding = nn.Linear(patch_len, self.embed_dim)

        if not query_share:
            self.query = nn.Parameter(
                torch.randn(
                    1, enc_in, self.past_patch_num + self.future_patch_num, self.embed_dim
                )
            )

        self.latent_cross_attention = AttentionBlock(
            n_heads, latent_dim, self.embed_dim, latent_d_ff, dropout
        )
        self.latent_attention_blocks = nn.ModuleList(
            [
                AttentionBlock(n_heads, latent_dim, latent_dim, latent_d_ff, dropout)
                for _ in range(3)
            ]
        )
        self.write_cross_attention = AttentionBlock(
            n_heads, self.embed_dim, latent_dim, d_ff, dropout
        )
        self.query_cross_attention = AttentionBlock(
            n_heads, self.embed_dim, self.embed_dim, d_ff, dropout
        )

        self.output_projection = nn.Linear(self.embed_dim, patch_len)

    def forecast(self, inputs):
        # RevIN-style instance normalisation
        means = inputs.mean(1, keepdim=True).detach()
        inputs = inputs - means
        stdev = torch.sqrt(
            torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        inputs = inputs / stdev

        # Patching (B, S, C) -> (B, C, P_N, D)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.unfold(2, self.patch_size, self.patch_size)
        batch_size, in_channels, patch_num, _ = inputs.size()
        inputs = self.patch_embedding(inputs)

        # Add patch (time) + channel positional embeddings
        inputs = inputs + self.patch_positional_embedding[:, :, : self.past_patch_num, :]
        inputs = inputs + self.channel_positional_embedding
        inputs = inputs.view(batch_size, in_channels * patch_num, self.embed_dim)

        # Latent cross/self attention (Perceiver core)
        if self.use_latent:
            latent = self.latent_array.expand(batch_size, -1, -1)
            for _ in range(self.num_latent_blocks):
                latent = self.latent_cross_attention(latent, inputs)
                for block in self.latent_attention_blocks:
                    latent = block(latent, latent)
                inputs = self.write_cross_attention(inputs, latent)

        # Build the future query
        if self.query_share:
            query = (
                self.patch_positional_embedding[:, :, self.past_patch_num :, :]
                + self.channel_positional_embedding
            )
        else:
            query = self.query[:, :, self.past_patch_num :, :]

        query = (
            query.expand(batch_size, -1, -1, -1)
            .contiguous()
            .reshape(batch_size * in_channels, -1, self.embed_dim)
        )
        inputs = (
            inputs.view(batch_size, in_channels, patch_num, self.embed_dim)
            .contiguous()
            .reshape(batch_size * in_channels, -1, self.embed_dim)
        )

        outputs = self.query_cross_attention(query, inputs)
        outputs = outputs.reshape(batch_size, in_channels, -1, self.embed_dim)
        outputs = self.output_projection(outputs)
        outputs = outputs.view(batch_size, in_channels, -1).contiguous().permute(0, 2, 1)

        # De-normalise
        outputs = outputs * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        outputs = outputs + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return outputs

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, C]
