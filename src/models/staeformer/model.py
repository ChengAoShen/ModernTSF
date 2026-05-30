"""ModernTSF adapter for the STAEformer spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STAEformer), Apache-2.0.

STAEformer (CIKM 2023) — "Spatio-Temporal Adaptive Embedding Makes Vanilla
Transformer SOTA for Traffic Forecasting" (https://arxiv.org/abs/2308.10425).
It augments a vanilla transformer with a learnable *adaptive* node/time
embedding plus time-of-day / day-of-week embeddings, then alternates temporal
and spatial self-attention. The adaptive embedding makes a predefined
adjacency matrix unnecessary, so ``adj_mx`` is accepted but unused.

The upstream BasicTS arch expects ``history_data`` of shape
``(B, L, N, input_dim)`` where channel 0 is the value, channel 1 is the
*time-of-day* index (normalised to ``[0, 1)``) and channel 2 is the
*day-of-week* index (a raw integer ``0..6``). This adapter rebuilds that
layout from ModernTSF's ``(x_enc, x_mark_enc)`` via ``to_spatiotemporal`` and
returns ``(B, pred_len, N)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal


# ---------------------------------------------------------------------------
# Vendored upstream layers (device-agnostic; no hardcoded cuda).
# ---------------------------------------------------------------------------
class AttentionLayer(nn.Module):
    """Multi-head attention across the ``-2`` dim (``-1`` dim is ``model_dim``)."""

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)

        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    """Upstream BasicTS STAEformer architecture (device-agnostic)."""

    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ):
        # history_data: (B, in_steps, num_nodes, input_dim + tod + dow)
        x = history_data
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))

        return out  # (B, out_steps, num_nodes, output_dim)


# ---------------------------------------------------------------------------
# ModernTSF adapter.
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """Adapter wrapping the upstream STAEformer architecture.

    Parameters
    ----------
    seq_len : int
        Input sequence length (``in_steps``).
    pred_len : int
        Forecast horizon (``out_steps``).
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray, optional
        ``(N, N)`` adjacency. STAEformer relies on learnable adaptive
        embeddings instead of a predefined graph, so this is accepted (and
        injected by the runner) but unused.
    input_dim : int
        Number of input channels fed to the backbone. STAEformer uses the
        value plus its time-of-day / day-of-week indices, so this defaults to
        3; only the first ``input_dim`` channels reach ``input_proj``.
    steps_per_day : int
        Time-of-day vocabulary size (24 for hourly smoke data).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        steps_per_day: int = 24,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_mixed_proj: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        # The history tensor built by ``to_spatiotemporal`` carries the value
        # plus two calendar channels [time_in_day, day_in_week], i.e. 3 chans.
        # STAEformer's ``input_proj`` only consumes the first ``input_dim``
        # channels (the value when input_dim == 1), while tod/dow are read from
        # fixed positions 1 and 2. Keep input_dim <= 3.
        self.net = STAEformer(
            num_nodes=num_nodes,
            in_steps=seq_len,
            out_steps=pred_len,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            output_dim=1,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=use_mixed_proj,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Either node-structured covariates ``(B, seq_len, N, F)`` (with
            ``F == 2`` calendar channels ``[time_in_day, day_in_week]``) or raw
            ``(B, seq_len, 6)`` calendar stamps.
        x_dec, x_mark_dec, mask
            Unused.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)

        # STAEformer indexes the day-of-week embedding table with raw integer
        # weekday indices (0..6), but ModernTSF's calendar covariates carry
        # ``day_in_week = weekday / 7`` in [0, 1). Rescale channel 2 back to a
        # raw index so ``dow.long()`` lands in the valid 0..6 range. Channel 1
        # (time_in_day in [0, 1)) is already what the backbone expects (it
        # multiplies by ``steps_per_day`` internally).
        if history.shape[-1] > 2:
            history = history.clone()
            history[..., 2] = history[..., 2] * 7.0

        out = self.net(
            history,
            None,
            batch_seen=0,
            epoch=0,
            train=self.training,
        )  # (B, pred_len, N, 1)

        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
