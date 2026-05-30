"""ModernTSF adapter for the BigST spatio-temporal graph forecaster.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/BigST), Apache-2.0.

BigST (VLDB 2024) is a linear-complexity spatio-temporal GNN for very large
road networks. Upstream it has an optional two-stage design: a long-sequence
``BigSTPreprocess`` feature extractor is pre-trained on the full dataset and
its frozen features are concatenated into the main model (``use_long=True``).
That pre-training stage needs a dataset-scale corpus and an on-disk checkpoint,
which a tiny smoke bundle cannot provide, so this adapter vendors only the
single-stage path (``use_long=False``). In that mode BigST is a standalone
GNN driven by learned adaptive node embeddings plus a linearized spatial
convolution; it consumes the value channel and ``[time_in_day, day_in_week]``
calendar covariates and emits ``(B, pred_len, N, 1)``.

The adapter builds the ``(B, T, N, 1 + F)`` spatio-temporal tensor from
ModernTSF's ``(x_enc, x_mark_enc)`` via ``to_spatiotemporal`` (value in
channel 0, calendar covariates in 1..), transposes to the upstream
``(B, N, T, D)`` layout, and squeezes the output channel back to
``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.bigst._upstream import Model as _BigST


class Model(nn.Module):
    """Adapter wrapping the single-stage upstream BigST model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray, optional
        ``(N, N)`` predefined adjacency injected by the runner. BigST's
        single-stage path learns its graph from adaptive node embeddings and
        does not require a predefined adjacency; when provided it is kept as a
        normalized buffer (``supports``) so a future spatial-regularization
        variant can use it, but it does not change the forward pass here.
    input_dim : int
        Number of channels per node in the spatio-temporal tensor
        (``1`` value + ``F`` calendar covariates). Defaults to ``3``.
    hid_dim, node_dim, time_dim : int
        Model widths (kept small for fast smoke runs).
    tod_size : int
        Time-of-day vocabulary size (samples per day).
    dow_size : int
        Day-of-week vocabulary size.
    tau, random_feature_dim, dropout
        Linearized-convolution hyper-parameters.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        hid_dim: int = 16,
        node_dim: int = 8,
        time_dim: int = 8,
        tod_size: int = 24,
        dow_size: int = 7,
        tau: float = 1.0,
        random_feature_dim: int = 16,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        self.net = _BigST(
            seq_num=seq_len,
            in_dim=input_dim,
            out_dim=pred_len,
            hid_dim=hid_dim,
            num_nodes=num_nodes,
            tau=tau,
            random_feature_dim=random_feature_dim,
            node_emb_dim=node_dim,
            time_emb_dim=time_dim,
            use_residual=use_residual,
            use_bn=use_bn,
            use_spatial=False,
            use_long=False,
            dropout=dropout,
            time_of_day_size=tod_size,
            day_of_week_size=dow_size,
            supports=None,
        )

        # Keep a normalized adjacency as a buffer (not on the forward path for
        # the single-stage variant). Registered lazily so device follows input.
        if adj_mx is not None:
            adj = np.asarray(adj_mx, dtype=np.float32)
            deg = adj.sum(axis=1, keepdims=True)
            deg[deg == 0.0] = 1.0
            adj = adj / deg
            self.register_buffer("adj_mx", torch.from_numpy(adj), persistent=False)
        else:
            self.adj_mx = None

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future node values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Either raw calendar stamps ``(B, seq_len, 6)`` or node-structured
            covariates ``(B, seq_len, N, F)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        # (B, T, N, 1 + F) — value in channel 0, calendar covariates after.
        history = to_spatiotemporal(x_enc, x_mark_enc)

        # Match the requested input_dim (slice or zero-pad the covariate tail).
        d = history.shape[-1]
        if d > self.input_dim:
            history = history[..., : self.input_dim]
        elif d < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - d)
            )
            history = torch.cat([history, pad], dim=-1)

        history = history.transpose(1, 2)  # (B, N, T, D) upstream layout
        out = self.net(history)["prediction"]  # (B, pred_len, N, 1)

        if out.dim() == 4:
            out = out[..., 0]
        return out  # (B, pred_len, N)
