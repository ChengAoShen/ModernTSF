"""ModernTSF adapter for the STID spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines / src/basicts/models/STID), Apache-2.0.

STID (CIKM 2022, https://arxiv.org/abs/2208.05233) is a simple MLP forecaster
that augments per-node series embeddings with learnable *spatial identity*
embeddings (one per node) and *temporal identity* embeddings indexed by
time-of-day and day-of-week. It needs no adjacency matrix — the learnable node
embeddings stand in for graph structure — so ``adj_mx`` is accepted but unused.

This adapter converts ModernTSF's ``(x_enc, x_mark_enc)`` into the upstream
``(inputs, inputs_timestamps)`` layout via
:func:`models._external.marks.to_spatiotemporal`, where channel 0 is the value
and the trailing channels are the normalized calendar features
``[time_in_day, day_in_week]`` in ``[0, 1)`` (used directly as embedding-table
indices). It returns ``(B, pred_len, N)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


class _ResMLPLayer(nn.Module):
    """MLP block with a residual connection (vendored from BasicTS)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.act_fn = getattr(F, hidden_act)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act_fn(self.fc1(inputs)))) + inputs


class _STID(nn.Module):
    """Vendored STID architecture (BasicTS), with plain kwargs.

    Forward takes ``inputs`` ``(B, L, N)`` and ``inputs_timestamps``
    ``(B, L, T)`` whose calendar channels are normalized to ``[0, 1)`` and used
    as embedding indices. Returns ``(B, output_len, N)``.
    """

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        input_hidden_size: int = 32,
        intermediate_size: int | None = None,
        hidden_act: str = "relu",
        num_layers: int = 1,
        if_spatial: bool = True,
        spatial_hidden_size: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        num_time_in_day: int = 24,
        num_day_in_week: int = 7,
        tid_hidden_size: int = 32,
        diw_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.input_hidden_size = input_hidden_size

        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.num_time_in_day = num_time_in_day
        self.num_day_in_week = num_day_in_week

        # spatial (node identity) embeddings
        if self.if_spatial:
            self.spatial_emb = nn.Parameter(
                torch.empty(num_features, spatial_hidden_size)
            )
            nn.init.xavier_uniform_(self.spatial_emb)
        # temporal identity embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(num_time_in_day, tid_hidden_size)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(num_day_in_week, diw_hidden_size)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Linear(self.input_len, self.input_hidden_size)

        # encoding
        self.hidden_size = (
            self.input_hidden_size
            + spatial_hidden_size * int(self.if_spatial)
            + tid_hidden_size * int(self.if_time_in_day)
            + diw_hidden_size * int(self.if_day_in_week)
        )
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else self.hidden_size
        )
        self.encoder = nn.Sequential(
            *[
                _ResMLPLayer(self.hidden_size, self.intermediate_size, hidden_act)
                for _ in range(num_layers)
            ]
        )

        # regression layer
        self.regression_layer = nn.Linear(self.hidden_size, self.output_len)

    def forward(
        self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor
    ) -> torch.Tensor:
        # Timestamps are normalized to [0, 1); rescale to integer indices.
        # Created index tensors live on the input's device (no hardcoded cuda).
        time_in_day_emb = (
            self.time_in_day_emb[
                (inputs_timestamps[:, -1, 0] * self.num_time_in_day)
                .long()
                .clamp(0, self.num_time_in_day - 1)
            ]
            if self.if_time_in_day
            else None
        )
        day_in_week_emb = (
            self.day_in_week_emb[
                (inputs_timestamps[:, -1, 1] * self.num_day_in_week)
                .long()
                .clamp(0, self.num_day_in_week - 1)
            ]
            if self.if_day_in_week
            else None
        )

        # time series embedding
        inputs = inputs.transpose(1, 2)  # [B, N, L]
        time_series_emb = self.time_series_emb_layer(inputs)  # [B, N, H]
        emb = [time_series_emb]

        if self.if_spatial:
            emb.append(self.spatial_emb.unsqueeze(0).expand(inputs.shape[0], -1, -1))
        if time_in_day_emb is not None:
            emb.append(time_in_day_emb.unsqueeze(1).expand(-1, inputs.shape[1], -1))
        if day_in_week_emb is not None:
            emb.append(day_in_week_emb.unsqueeze(1).expand(-1, inputs.shape[1], -1))

        hidden = torch.cat(emb, dim=-1)  # [B, N, hidden_size]
        hidden = self.encoder(hidden)  # [B, N, hidden_size]
        prediction = self.regression_layer(hidden).transpose(1, 2)  # [B, out, N]
        return prediction


class Model(nn.Module):
    """Adapter wrapping the vendored STID architecture.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N`` (injected by the runner from the dataset).
    adj_mx : np.ndarray, optional
        ``(N, N)`` adjacency, accepted for the graph-model interface but unused
        by STID (learnable node embeddings replace explicit graph structure).
    input_dim : int
        Number of input channels per node in the source data (value + calendar);
        retained for interface compatibility, not consumed directly.
    embed_dim : int
        Series / spatial / temporal embedding dimension.
    num_layers : int
        Number of residual-MLP encoder blocks.
    num_time_in_day : int
        Time-of-day vocabulary size (samples per day).
    num_day_in_week : int
        Day-of-week vocabulary size.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        embed_dim: int = 32,
        num_layers: int = 1,
        num_time_in_day: int = 24,
        num_day_in_week: int = 7,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        # adj_mx is unused by STID; kept only for the graph-model factory contract.
        self.net = _STID(
            input_len=seq_len,
            output_len=pred_len,
            num_features=num_nodes,
            input_hidden_size=embed_dim,
            num_layers=num_layers,
            if_spatial=True,
            spatial_hidden_size=embed_dim,
            if_time_in_day=if_time_in_day,
            if_day_in_week=if_day_in_week,
            num_time_in_day=num_time_in_day,
            num_day_in_week=num_day_in_week,
            tid_hidden_size=embed_dim,
            diw_hidden_size=embed_dim,
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
            Calendar covariates: node-structured ``(B, seq_len, N, F)`` or raw
            stamps ``(B, seq_len, 6)``.
        x_dec, x_mark_dec, mask
            Unused by STID.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        # (B, L, N, 1 + F): channel 0 value, then [time_in_day, day_in_week].
        history = to_spatiotemporal(x_enc, x_mark_enc)
        values = history[..., 0]  # (B, L, N)
        # Per-step calendar features, broadcast-identical across nodes; take node 0.
        timestamps = history[:, :, 0, 1:]  # (B, L, F)
        out = self.net(values, timestamps)  # (B, pred_len, N)
        return out
