"""ModernTSF adapter for the DGCRN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/DGCRN), Apache-2.0.

DGCRN (TKDD 2023) is a dynamic graph convolutional recurrent (GRU) network. It
needs a predefined adjacency (built here from the injected ``(N, N)`` ``adj_mx``)
and *additionally* learns a dynamic graph filter at every recurrent step from
node embeddings modulated by the hidden state.

ModernTSF feeds the model a value tensor ``(B, seq_len, N)`` plus node-
structured covariate marks ``(B, seq_len, N, F)`` (or raw calendar stamps).
This adapter reassembles the BasicTS spatiotemporal layout ``(B, L, N, 1 + F)``
(value channel 0, then calendar/covariate channels), supplies a zero-filled
future covariate block of the same channel layout (the upstream decoder reads
channel 1 — time-of-day — as its driving input), drives the upstream module with
the BasicTS forward signature, and squeezes the output channel back to
``(B, pred_len, N)``.

The predefined transition matrices are registered as device-following buffers;
no tensor is created on a hardcoded CUDA device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.dgcrn._upstream import DGCRN


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Row-normalised adjacency ``D^{-1} (A + I)`` (dense, float32)."""
    a = adj + np.eye(adj.shape[0], dtype=adj.dtype)
    d = a.sum(axis=1, keepdims=True)
    d = np.where(d > 0, d, 1.0)
    return (a / d).astype(np.float32)


class Model(nn.Module):
    """Adapter wrapping the upstream DGCRN architecture.

    Parameters
    ----------
    seq_len : int
        Input sequence length (number of encoder recurrent steps).
    pred_len : int
        Forecast horizon (number of decoder recurrent steps).
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray | None
        Predefined ``(N, N)`` adjacency, injected by the runner from the
        dataset. A row-normalised forward + reverse transition pair is built
        from it. When ``None`` an identity adjacency is used as a fallback.
    input_dim : int
        Number of input channels fed to the model. DGCRN ties the encoder and
        decoder channel counts together: the decoder step concatenates the
        previous 1-channel prediction with the 1-channel time-of-day feature, so
        ``input_dim`` must be ``2`` ( ``[value, time_in_day]`` ). Other values
        are clamped to ``2`` to keep the recurrent dimensions consistent.
    gcn_depth : int
        Graph-convolution propagation depth.
    rnn_size : int
        Hidden size of the recurrent cell.
    node_dim : int
        Node-embedding dimension.
    hyper_gnn_dim : int
        Hidden dimension of the dynamic (hyper) graph filter network.
    middle_dim : int
        Intermediate dimension of the hyper filter network.
    subgraph_size : int
        Top-k neighbour count for the learned graph (kept for parity).
    tanhalpha : float
        Saturation factor for the learned-adjacency tanh.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 2,
        gcn_depth: int = 1,
        rnn_size: int = 16,
        node_dim: int = 8,
        hyper_gnn_dim: int = 8,
        middle_dim: int = 2,
        subgraph_size: int = 20,
        tanhalpha: float = 3.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # DGCRN's decoder feeds [prev_prediction (1ch), time_of_day (1ch)] back
        # into the same recurrent cell as the encoder input, so the channel
        # count is fixed at 2 ([value, time_in_day]).
        input_dim = 2
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        if adj_mx is None:
            adj = np.eye(num_nodes, dtype=np.float32)
        else:
            adj = np.asarray(adj_mx, dtype=np.float32)
            adj = adj[:num_nodes, :num_nodes]

        # Predefined forward + reverse row-normalised transition matrices,
        # registered as device-following buffers.
        self.register_buffer(
            "_predefined_fwd", torch.from_numpy(_normalize_adj(adj))
        )
        self.register_buffer(
            "_predefined_bwd", torch.from_numpy(_normalize_adj(adj.T))
        )

        self.net = DGCRN(
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            predefined_A=None,  # set per-forward from the buffers (device-correct)
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            middle_dim=middle_dim,
            seq_length=seq_len,
            in_dim=input_dim,
            tanhalpha=tanhalpha,
            rnn_size=rnn_size,
            hyperGNN_dim=hyper_gnn_dim,
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
            Node-structured covariate marks ``(B, seq_len, N, F)`` or raw
            calendar stamps ``(B, seq_len, 6)``.
        x_dec, x_mark_dec, mask
            Unused (no teacher forcing / future leakage at inference).

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        history = self._coerce_channels(history)

        # The upstream decoder reads future channel 1 (time-of-day) as its
        # driving input. We have no future targets, so build a zero future
        # block with the same channel layout and horizon length.
        b = history.shape[0]
        future = history.new_zeros((b, self.pred_len, self.num_nodes, self.input_dim))

        # The buffers carry the correct device/dtype after ``.to(device)``.
        self.net.predefined_A = [self._predefined_fwd, self._predefined_bwd]

        out = self.net(
            history,
            future,
            batch_seen=0,
            epoch=0,
            train=self.training,
            task_level=self.pred_len,
        )  # (B, pred_len, N, 1)
        return out[..., 0]

    def _coerce_channels(self, history: torch.Tensor) -> torch.Tensor:
        """Pad / truncate the channel axis to ``self.input_dim``."""
        c = history.shape[-1]
        if c == self.input_dim:
            return history
        if c > self.input_dim:
            return history[..., : self.input_dim]
        pad = history.new_zeros((*history.shape[:-1], self.input_dim - c))
        return torch.cat([history, pad], dim=-1)
