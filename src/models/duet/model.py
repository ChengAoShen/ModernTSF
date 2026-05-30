"""DUET model implementation.

Vendored/adapted from https://github.com/decisionintelligence/DUET
(ts_benchmark/baselines/duet/models/duet_model.py and its layers/utils),
MIT License (Copyright (c) 2024 Huawei Technologies Co., Ltd).

DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting (KDD 2025).

Adapted for ModernTSF: the upstream ``config``-object constructor is replaced
with plain keyword arguments, and the core architecture is wired into the
ModernTSF ``(B, T, C)`` forward contract returning ``(B, pred_len, c_out)``.
The shared ``series_decomp`` (``models.module.autoformer_encdec``) and ``RevIN``
(``models.module.revin``) layers are reused. The DUET-specific blocks --
the channel-clustering Mahalanobis mask, the masked channel Transformer encoder
(whose ``FullAttention`` applies a learned soft channel mask), the linear
pattern extractor experts, the distributional router, and the sparsely-gated
mixture-of-experts cluster -- are kept local to this file because their
signatures differ from the shared same-name blocks.
"""

from __future__ import annotations

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions.normal import Normal
from torch.nn.functional import gumbel_softmax

from models.module.autoformer_encdec import series_decomp
from models.module.revin import RevIN


# --------------------------------------------------------------------------- #
# Distributional router (gate / noise encoder)
# --------------------------------------------------------------------------- #
class DistributionalRouter(nn.Module):
    def __init__(self, seq_len, hidden_size, num_experts):
        super().__init__()
        self.distribution_fit = nn.Sequential(
            nn.Linear(seq_len, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts, bias=False),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        return self.distribution_fit(mean)


# --------------------------------------------------------------------------- #
# Linear pattern extractor expert (decomposition linear)
# --------------------------------------------------------------------------- #
class LinearExtractor(nn.Module):
    def __init__(self, seq_len, d_model, moving_avg, enc_in, CI):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = d_model
        self.decompsition = series_decomp(moving_avg)
        self.channels = enc_in
        self.enc_in = 1 if CI else enc_in

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.pred_len, self.enc_in)).to(x_enc.device)
        dec_out = self.encoder(x_enc)
        return dec_out[:, -self.pred_len :, :]


# --------------------------------------------------------------------------- #
# Sparsely-gated mixture-of-experts cluster
# --------------------------------------------------------------------------- #
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class LinearExtractorCluster(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        moving_avg,
        enc_in,
        CI,
        num_experts,
        k,
        hidden_size,
        noisy_gating,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = seq_len
        self.k = k
        self.experts = nn.ModuleList(
            [
                LinearExtractor(seq_len, d_model, moving_avg, enc_in, CI)
                for _ in range(num_experts)
            ]
        )
        self.W_h = nn.Parameter(torch.eye(num_experts))
        self.gate = DistributionalRouter(seq_len, hidden_size, num_experts)
        self.noise = DistributionalRouter(seq_len, hidden_size, num_experts)

        self.n_vars = enc_in
        self.revin = RevIN(self.n_vars)

        self.CI = CI
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = self.gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        if self.CI:
            x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)
            x_norm = self.revin(x_norm, "norm")
            x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)
        else:
            x_norm = self.revin(x, "norm")

        expert_inputs = dispatcher.dispatch(x_norm)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss


# --------------------------------------------------------------------------- #
# Masked channel-attention Transformer (DUET-specific FullAttention masking)
# --------------------------------------------------------------------------- #
class FullAttention(nn.Module):
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

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0)
            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


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
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(
            torch.randn(frequency_size, frequency_size), requires_grad=True
        )

    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)
        diff = X1 - X2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)
        exp_dist = 1 / (dist + 1e-10)

        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()

        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)
        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99
        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(
            resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d
        )
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)
        sample = self.bernoulli_gumbel_rsample(p)
        mask = sample.unsqueeze(1)
        return mask


# --------------------------------------------------------------------------- #
# DUET model
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_ff=2048,
        dropout=0.1,
        fc_dropout=0.1,
        factor=3,
        activation="gelu",
        moving_avg=25,
        num_experts=4,
        k=2,
        hidden_size=256,
        noisy_gating=True,
        CI=True,
        output_attention=False,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.CI = CI
        self.n_vars = enc_in

        self.cluster = LinearExtractorCluster(
            seq_len=seq_len,
            d_model=d_model,
            moving_avg=moving_avg,
            enc_in=enc_in,
            CI=CI,
            num_experts=num_experts,
            k=k,
            hidden_size=hidden_size,
            noisy_gating=noisy_gating,
        )
        self.mask_generator = Mahalanobis_mask(seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.linear_head = nn.Sequential(
            nn.Linear(d_model, pred_len), nn.Dropout(fc_dropout)
        )

    def _forecast(self, x):
        # x: [B, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(x, "b l n -> (b n) l 1")
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=x.shape[0]
            )
        else:
            temporal_feature, L_importance = self.cluster(x)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")
        if self.n_vars > 1:
            changed_input = rearrange(x, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)
            channel_group_feature, _ = self.Channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )
            output = self.linear_head(channel_group_feature)
        else:
            output = self.linear_head(temporal_feature)

        output = rearrange(output, "b n d -> b d n")
        output = self.cluster.revin(output, "denorm")
        return output, L_importance

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out, _ = self._forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
