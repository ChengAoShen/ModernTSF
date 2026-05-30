"""DTAF model implementation.

Vendored/adapted from https://github.com/decisionintelligence/DTAF
(ts_benchmark/baselines/dtaf/model/DTAF_model.py and
ts_benchmark/baselines/dtaf/layer/{Embed,Linear_extractor,kan}.py).

DTAF: "Towards Non-Stationary Time Series Forecasting with Temporal
Stabilization and Frequency Differencing" (AAAI 2026, arXiv:2511.08229).

License note: the DTAF repository ships no explicit LICENSE file. It is
published by the decisionintelligence group as a baseline built on top of the
MIT-licensed TFB benchmark (https://github.com/decisionintelligence/TFB) and
lives in the TFB-derived ``ts_benchmark/baselines`` tree; the parent TFB
framework is MIT-licensed.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, the non-forecasting branches are dropped, and the
shared layers ``PatchEmbedding`` (models.module.embed) and ``series_decomp``
(models.module.autoformer_encdec) are reused. The dual-branch TFS (temporal
stabilizing fusion via a non-stationary KAN mixture-of-experts) and FWM
(frequency wave modeling via frequency differencing) blocks are kept local to
this file. The upstream ``torch.save`` debug side effects and the auxiliary
``stables`` return value are removed so ``forward`` returns a single
``(B, pred_len, c_out)`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.autoformer_encdec import series_decomp
from models.module.embed import PatchEmbedding
from models.dtaf.kan import KAN, KANLinear


class Expert(nn.Module):
    def __init__(self, input_dim, div):
        super().__init__()
        self.network = KAN(layers_hidden=[input_dim, input_dim // div, input_dim])

    def forward(self, x):
        return self.network(x)


class MOE(nn.Module):
    def __init__(self, expert_num, input_dim, div):
        super().__init__()
        self.experts = nn.ModuleList()
        self.router = KANLinear(input_dim, expert_num)
        for _ in range(expert_num):
            self.experts.append(Expert(input_dim=input_dim, div=div))

    def forward(self, x):
        router = self.router(x).softmax(-1)
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.einsum("bpn,bpnd->bpd", router, experts_out)


class LinearExtractor(nn.Module):
    """DLinear-style decomposition extractor operating on the d_model axis.

    Upstream ``Linear_extractor`` uses ``seq_len == pred_len == d_model``; we
    keep that contract and reuse the shared ``series_decomp``.
    """

    def __init__(self, d_model, moving_avg):
        super().__init__()
        self.decompsition = series_decomp(moving_avg)
        self.Linear_Seasonal = nn.Linear(d_model, d_model)
        self.Linear_Trend = nn.Linear(d_model, d_model)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / d_model) * torch.ones([d_model, d_model])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / d_model) * torch.ones([d_model, d_model])
        )

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        return seasonal_output + trend_output


class TFS(nn.Module):
    """Temporal Stabilizing Fusion block."""

    def __init__(
        self, input_dim, patch_num, dropout, moving_avg, expert_num, kan_div,
        aggregated_norm
    ):
        super().__init__()
        self.expert_num = expert_num
        self.aggregated_norm = aggregated_norm
        self.MLP = nn.Linear(input_dim, input_dim)
        self.extractor_his = LinearExtractor(input_dim, moving_avg)
        self.weight_linear = nn.Linear(input_dim, patch_num)
        self.dropout = nn.Dropout(dropout)
        self.extractor_cur = LinearExtractor(input_dim, moving_avg)
        self.gate = nn.Linear(input_dim, 1)
        if self.aggregated_norm == 1:
            self.norm = nn.LayerNorm(input_dim)
        if expert_num > 0:
            self.moe = MOE(expert_num=expert_num, input_dim=input_dim, div=kan_div)

    def forward(self, x):
        origin = x
        if self.expert_num > 0:
            x = x - self.moe(x)
        H = self.extractor_his(x)
        weight_current = self.gate(self.extractor_cur(origin)).repeat(
            1, 1, origin.shape[-1]
        )
        weight = self.weight_linear(H).softmax(dim=-1)

        adj = torch.tril(weight, diagonal=0)
        aggregated = torch.matmul(adj, x)

        H_history = self.dropout(self.MLP(aggregated))
        H_current = self.dropout(weight_current) * x

        if self.aggregated_norm == 1:
            out = self.norm(H_history + H_current)
        else:
            out = H_history + H_current
        return out


class Attention(nn.Module):
    def __init__(self, extra_d, heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=extra_d, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(extra_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x


class Predict(nn.Module):
    def __init__(self, nf, target_window, dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        d_model=32,
        e_layers=1,
        patch_len=16,
        stride=8,
        heads=2,
        dropout=0.1,
        moving_avg=25,
        expert_num=2,
        kan_div=4,
        k=1,
        aggregated_norm=1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.e_layers = e_layers
        self.k = k

        self.patch_num = int((seq_len - patch_len) / stride + 2)

        self.TFSs = nn.ModuleList(
            [
                TFS(
                    d_model,
                    self.patch_num,
                    dropout,
                    moving_avg,
                    expert_num,
                    kan_div,
                    aggregated_norm,
                )
                for _ in range(e_layers)
            ]
        )
        self.predictor = Predict(2 * d_model * self.patch_num, pred_len, dropout)
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, stride, dropout
        )
        self.temporal_attention = Attention(d_model, heads, dropout)
        self.frequency_attention = Attention(d_model, heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _get_mean_std(self, x):
        means = x.mean(1, keepdim=True)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev

    def forecast(self, x_enc):
        B, L, D = x_enc.size()

        # Instance Norm
        x_enc, means, stdev = self._get_mean_std(x_enc)

        # Patch & Embedding -> (B*D, patch_num, d_model)
        enc_out, _ = self.patch_embedding(x_enc.transpose(1, 2))

        # TFS
        enc_out_TFS = enc_out
        for i in range(self.e_layers):
            agg = self.TFSs[i](enc_out_TFS)
            enc_out_TFS = self.norm(self.drop(agg) + enc_out_TFS)

        # FWM (frequency wave modeling via frequency differencing)
        enc_out = enc_out_TFS
        H_t = enc_out
        wave = torch.zeros(
            enc_out.shape[0], enc_out.shape[1], enc_out.shape[2] // 2 + 1,
            device=enc_out.device,
        )
        freq = torch.fft.rfft(enc_out)
        wave[:, 1:, :] = torch.exp(
            torch.abs(freq[:, 1:, :]) - torch.abs(freq[:, :-1, :])
        )

        k = min(self.k, wave.shape[-1])
        _, topk_indices = torch.topk(wave, k, dim=-1)
        mask = torch.zeros_like(freq, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, value=True)

        filtered_freq = torch.where(mask, freq, torch.zeros_like(freq))
        H_f = torch.fft.irfft(filtered_freq, n=enc_out.shape[-1])
        H_f[:, 0, :] = enc_out[:, 0, :]

        # dual-attention
        H_f = self.frequency_attention(H_f)
        H_t = self.frequency_attention(H_t)
        enc_out = torch.cat([H_t, H_f], dim=-2)

        enc_out = torch.reshape(
            enc_out, (-1, D, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)
        out = self.predictor(enc_out)
        out = out.permute(0, 2, 1)
        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]
