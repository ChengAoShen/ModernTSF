"""ModernTSF adapter for the D2STGNN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/D2STGNN), Apache-2.0.

D2STGNN (VLDB 2022) is a decoupled dynamic spatial-temporal graph neural
network. It separates the diffusion (spatial) and inherent (temporal) signals
into two interacting branches, building a *dynamic* graph from the running
hidden state on top of a *static* learned graph and a *predefined* adjacency.

ModernTSF feeds the model a value tensor ``(B, seq_len, N)`` plus
node-structured covariate marks ``(B, seq_len, N, F)`` (or raw calendar
stamps). This adapter reassembles the BasicTS spatiotemporal layout
``(B, L, N, 1 + F)`` — value in channel 0, then ``[time_in_day, day_in_week]``
calendar covariates — drives the upstream module with the BasicTS forward
signature, and squeezes the output channel back to ``(B, pred_len, N)``.

The injected ``(N, N)`` ``adj_mx`` is converted into the two random-walk
transition matrices (the ``"doubletransition"`` form used by the original
recipe) and registered as buffers so they follow the model's device; no tensor
is created on a hardcoded CUDA device.

Note: D2STGNN ties the distance-function input length and the forecast horizon
to a single ``seq_length`` argument, so this adapter requires
``seq_len == pred_len``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.d2stgnn._upstream import D2STGNN


def _transition_matrix(adj: np.ndarray) -> np.ndarray:
    """Random-walk transition matrix ``P = D^{-1} A`` (DCRNN / GWNet form)."""
    row_sum = adj.sum(axis=1)
    d_inv = np.power(row_sum, -1.0, where=row_sum != 0)
    d_inv[np.isinf(d_inv)] = 0.0
    d_inv[row_sum == 0] = 0.0
    return (np.diag(d_inv) @ adj).astype(np.float32)


def _double_transition(adj: np.ndarray) -> list[np.ndarray]:
    """The ``"doubletransition"`` adjacency: forward + backward transitions."""
    return [_transition_matrix(adj), _transition_matrix(adj.T)]


class Model(nn.Module):
    """Adapter wrapping the upstream D2STGNN architecture.

    Parameters
    ----------
    seq_len : int
        Input sequence length. Must equal ``pred_len`` (D2STGNN ties the
        distance-function input length to the forecast horizon).
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray | None
        Predefined ``(N, N)`` adjacency, injected by the runner from the
        dataset. When ``None`` an identity matrix is used so the dynamic-graph
        mask remains well defined.
    input_dim : int
        Number of input channels reassembled for the network: 1 value channel
        plus ``input_dim - 1`` calendar covariates ``[time_in_day, day_in_week]``.
    num_feat : int
        Number of value features (channel 0..num_feat). Default 1.
    num_hidden : int
        Hidden embedding dimension.
    node_hidden : int
        Node-embedding dimension.
    time_emb_dim : int
        Time-of-day / day-of-week embedding dimension.
    k_s : int
        Spatial diffusion order.
    k_t : int
        Temporal kernel size for the localized ST convolution.
    gap : int
        Auto-regression gap; ``pred_len`` must be divisible by ``gap``.
    num_layers : int
        Number of decouple layers.
    dropout : float
        Dropout rate.
    time_in_day_size : int
        Time-of-day vocabulary size (number of slots per day).
    day_in_week_size : int
        Day-of-week vocabulary size.
    forecast_dim, output_hidden : int
        Forecast-branch and output-MLP hidden widths.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        num_feat: int = 1,
        num_hidden: int = 16,
        node_hidden: int = 8,
        time_emb_dim: int = 8,
        k_s: int = 2,
        k_t: int = 3,
        gap: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_in_day_size: int = 288,
        day_in_week_size: int = 7,
        forecast_dim: int = 64,
        output_hidden: int = 128,
    ) -> None:
        super().__init__()
        if seq_len != pred_len:
            raise ValueError(
                "D2STGNN ties its distance-function input length to the "
                f"forecast horizon; seq_len ({seq_len}) must equal pred_len "
                f"({pred_len})."
            )
        if pred_len % gap != 0:
            raise ValueError(
                f"pred_len ({pred_len}) must be divisible by gap ({gap})."
            )

        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_feat = num_feat

        # Predefined adjacency -> two random-walk transition matrices, stored as
        # buffers so they follow the model device. Identity fallback keeps the
        # dynamic-graph mask well defined when no adj is supplied.
        if adj_mx is None:
            adj_mx = np.eye(num_nodes, dtype=np.float32)
        adj = np.asarray(adj_mx, dtype=np.float32)
        self._adj_keys: list[str] = []
        for i, a in enumerate(_double_transition(adj)):
            key = f"adj_{i}"
            self.register_buffer(key, torch.from_numpy(a))
            self._adj_keys.append(key)
        adjs = [getattr(self, k) for k in self._adj_keys]

        model_args = dict(
            num_feat=num_feat,
            num_hidden=num_hidden,
            node_hidden=node_hidden,
            time_emb_dim=time_emb_dim,
            seq_length=pred_len,
            num_nodes=num_nodes,
            k_s=k_s,
            k_t=k_t,
            gap=gap,
            num_layers=num_layers,
            dropout=dropout,
            time_in_day_size=time_in_day_size,
            day_in_week_size=day_in_week_size,
            forecast_dim=forecast_dim,
            output_hidden=output_hidden,
            adjs=adjs,
        )
        self.net = D2STGNN(**model_args)

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
            Node-structured covariate marks ``(B, seq_len, N, F)`` or raw
            calendar stamps ``(B, seq_len, 6)``.
        x_dec, x_mark_dec, mask
            Unused by D2STGNN.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        # Keep value channel(s) + the calendar covariates the network expects.
        history = history[..., : self.input_dim]
        if history.shape[-1] < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - history.shape[-1])
            )
            history = torch.cat([history, pad], dim=-1)

        out = self.net(
            history,
            None,
            batch_seen=0,
            epoch=0,
            train=self.training,
        )  # (B, pred_len, N, 1)

        if out.dim() == 4:
            out = out[..., 0]
        return out
