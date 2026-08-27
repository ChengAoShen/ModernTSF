"""ModernTSF adapter for CRIB (Consistency-Regularized Information Bottleneck).

*CRIB: forecasting directly from partially observed multivariate series.*
Upstream: https://github.com/Muyiiiii/CRIB

CRIB patches the input, encodes it with a TCN + unified-variate Transformer into
an IB latent, and forecasts with a small MLP head. Its training objective has
three terms (paper Eq.)::

    L = IB_weight   * MAE(Ŷ, Y)                         # prediction
      + Consis_weight * MSE(enc_clean, enc_noisy)        # consistency regularizer
      + KL_weight   * KL( q(z|x) || N(0, I) )            # IB compression

The consistency and KL terms are computed **inside** ``forward`` from the input
alone (a noisy second view ``x + 0.01·N(mean,std)`` is built internally), so they
need no future target and ride the trainer's existing ``aux_loss`` convention.
The prediction term is supplied by ModernTSF's configured loss (use ``mae`` for
the paper recipe; ``IB_weight`` is implicitly 1.0).

This is a model-only port (per request): the upstream missing-value masking /
augmentation data pipeline is NOT included — CRIB simply trains on the standard
complete forecasting windows (equivalent to upstream ``missing_rate=0``). The
vendored core reproduces the upstream architecture (dead/unused submodules
dropped); only the patching adapter + ModernTSF interface are new.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.parametrizations import weight_norm


# --------------------------------------------------------------------------- #
# Vendored upstream modules (CRIB_utils / CRIB_embedding / CRIB_module / CRIB),
# trimmed of submodules that are instantiated but never used in forward.
# --------------------------------------------------------------------------- #
class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1), : x.size(2)]


class _RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise NotImplementedError


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(in_channel, out_channel, (1, kernel_size), stride, (0, padding), dilation)
        )
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv2d(out_channel, out_channel, (1, kernel_size), stride, (0, padding), dilation)
        )
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class _TCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel_list, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(len(out_channel_list)):
            dilation = 2 ** i
            ich = in_channel if i == 0 else out_channel_list[i - 1]
            layers.append(
                _TemporalBlock(
                    ich, out_channel_list[i], kernel_size, 1, dilation,
                    (kernel_size - 1) * dilation, dropout,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class _AttentionLayer(nn.Module):
    """Upstream quirk preserved: Q, K, V all go through ``query_projection``."""

    def __init__(self, attention, model_dim, heads_num):
        super().__init__()
        d = model_dim // heads_num
        self.inner_attention = attention
        self.query_projection = nn.Linear(model_dim, d * heads_num)
        self.out_projection = nn.Linear(d * heads_num, model_dim)
        self.heads_num = heads_num

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.heads_num
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.query_projection(keys).view(B, S, H, -1)
        values = self.query_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_projection(out)


class _EncoderLayer(nn.Module):
    def __init__(self, attention, model_dim, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = 4 * model_dim
        self.attention = attention
        self.conv1 = nn.Conv1d(model_dim, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, model_dim, 1)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _TransformerEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm_layer = norm_layer

    def forward(self, x):
        for layer in self.attn_layers:
            x = layer(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class _CRIBEncoder(nn.Module):
    def __init__(self, args, patch_num):
        super().__init__()
        self.args = args
        self.patch_num = patch_num
        self.softplus = nn.Softplus()
        self.enc_embedding_2 = _TCNBlock(
            in_channel=args.patch_len, out_channel_list=[64, args.model_dim],
            kernel_size=3, dropout=args.dropout,
        )
        self.encoder = _TransformerEncoder(
            attn_layers=[
                _EncoderLayer(
                    _AttentionLayer(
                        _Attention(scale=None, attention_dropout=args.dropout),
                        args.model_dim, args.heads_num,
                    ),
                    args.model_dim, args.dropout, args.activation,
                )
                for _ in range(args.enc_num)
            ],
            norm_layer=nn.LayerNorm(args.model_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(patch_num * args.model_dim, args.model_dim),
            nn.ReLU(),
            nn.Linear(args.model_dim, args.model_dim * 2),
        )

    def forward(self, x_enc):
        B, P, N, L = x_enc.shape
        x_enc = x_enc.permute(0, 3, 2, 1)
        enc_out = self.enc_embedding_2(x_enc)
        enc_out = enc_out.permute(0, 3, 2, 1).reshape(B, -1, self.args.model_dim)
        enc_out = self.encoder(enc_out)
        tmp = enc_out.reshape(B, P, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        mapped = self.projector(tmp)
        loc = mapped[:, :, : self.args.model_dim]
        scale = self.softplus(mapped[:, :, self.args.model_dim:]) + 1e-9
        dist = MultivariateNormal(loc=loc, covariance_matrix=torch.diag_embed(scale))
        return enc_out, dist


class _CRIBPredHead(nn.Module):
    def __init__(self, args, patch_num):
        super().__init__()
        self.args = args
        self.prediction_1 = nn.Linear(patch_num * args.model_dim, args.model_dim)
        self.act_1 = nn.ReLU()
        self.prediction_2 = nn.Linear(args.model_dim, args.pred_len)

    def forward(self, x_pred):
        B = x_pred.shape[0]
        x_pred = x_pred.reshape(B, -1, self.args.var_num, self.args.model_dim)
        x_pred = x_pred.permute(0, 2, 1, 3).reshape(B, self.args.var_num, -1)
        x_pred = self.act_1(self.prediction_1(x_pred))
        x_pred = self.prediction_2(x_pred)
        return x_pred.permute(0, 2, 1)  # (B, pred_len, var_num)


class _CRIBCore(nn.Module):
    """Verbatim CRIB forward (device taken from the input; dead code dropped)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_num = args.patch_num
        self.enc_pos_emded = _PositionalEmbedding(
            d_model=(args.patch_len + 1) // 2 * 2, max_len=5000
        )
        self.encoder = _CRIBEncoder(args, args.patch_num)
        self.predictor = _CRIBPredHead(args, args.patch_num)
        self.revinlayer = _RevIN(num_features=1)

    def forward(self, x, test_flag=False):
        B, P, N, L = x.shape
        x_1 = x.permute(0, 1, 3, 2).reshape(B, P * L, N)
        x_1 = self.revinlayer(x_1, mode="norm")
        x_1 = x_1.reshape(B, P, L, N).permute(0, 1, 3, 2).reshape(B, P * N, L)
        x_1 = x_1 + self.enc_pos_emded(x_1)
        x_1 = x_1.reshape(B, P, N, L)

        pz = MultivariateNormal(
            loc=torch.zeros(self.args.model_dim, device=x.device),
            covariance_matrix=torch.eye(self.args.model_dim, device=x.device),
        )
        enc_out_1, qz_x_1 = self.encoder(x_1)
        if test_flag:
            enc_out_2 = None
        else:
            noise = 0.01 * torch.normal(
                x_1.mean().item(), x_1.std().item(), x_1.shape, device=x.device
            )
            enc_out_2, _ = self.encoder(x_1 + noise)

        kl = torch.distributions.kl.kl_divergence(qz_x_1, pz)
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        kl = torch.sum(kl)

        preds = self.predictor(enc_out_1)
        preds = self.revinlayer(preds, mode="denorm")
        return enc_out_1, enc_out_2, preds, kl


class Model(nn.Module):
    """CRIB forecasting model.

    Parameters
    ----------
    seq_len, pred_len, enc_in : int
        Task dimensions. ``patch_len`` must divide ``seq_len``; ``model_dim``
        must be divisible by ``heads_num``.
    patch_len, model_dim, heads_num, enc_num, dropout, activation
        Backbone hyper-parameters.
    consis_weight, kl_weight : float
        Weights of the consistency (MSE) and IB (KL) regularizers, added to the
        configured prediction loss via ``aux_loss``. Upstream defaults 1.0 / 1e-6.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 8,
        model_dim: int = 32,
        heads_num: int = 4,
        enc_num: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        consis_weight: float = 1.0,
        kl_weight: float = 1e-6,
    ) -> None:
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(
                f"CRIB requires patch_len ({patch_len}) to divide seq_len ({seq_len})."
            )
        if model_dim % heads_num != 0:
            raise ValueError(
                f"CRIB requires model_dim ({model_dim}) divisible by heads_num ({heads_num})."
            )
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.consis_weight = float(consis_weight)
        self.kl_weight = float(kl_weight)
        args = SimpleNamespace(
            patch_len=patch_len,
            patch_num=self.patch_num,
            model_dim=model_dim,
            pred_len=pred_len,
            var_num=enc_in,
            dropout=dropout,
            heads_num=heads_num,
            enc_num=enc_num,
            activation=activation,
        )
        self.core = _CRIBCore(args)
        self.aux_loss: torch.Tensor | None = None

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        # (B, seq_len, N) -> (B, patch_num, N, patch_len)
        B, T, N = x.shape
        return (
            x.permute(0, 2, 1)
            .reshape(B, N, self.patch_num, self.patch_len)
            .permute(0, 2, 1, 3)
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        self.aux_loss = None
        enc_out_1, enc_out_2, preds, kl = self.core(
            self._patch(x), test_flag=not self.training
        )
        if self.training and enc_out_2 is not None:
            consis = F.mse_loss(enc_out_1, enc_out_2)
            self.aux_loss = self.consis_weight * consis + self.kl_weight * kl
        return preds
