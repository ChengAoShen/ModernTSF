"""ModernTSF adapter for the DCRNN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/DCRNN), Apache-2.0.

DCRNN (ICLR 2018) is a diffusion-convolutional recurrent (GRU) sequence-to-
sequence model. It REQUIRES a predefined adjacency: the diffusion convolution
uses *dual random-walk* transition matrices built from the injected ``(N, N)``
``adj_mx`` (forward random walk ``D_O^{-1} W`` and reverse ``D_I^{-1} W^T``).

ModernTSF feeds the model a value tensor ``(B, seq_len, N)`` plus node-
structured covariate marks ``(B, seq_len, N, F)`` (or raw calendar stamps).
This adapter reassembles the BasicTS spatiotemporal layout
``(B, L, N, 1 + F)`` (value channel 0, then calendar/covariate channels),
drives the upstream module with the BasicTS forward signature, and squeezes the
output channel back to ``(B, pred_len, N)``.

The transition matrices are registered as device-following buffers inside the
upstream cells; no tensor is created on a hardcoded CUDA device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.dcrnn._upstream import DCRNN


def _calculate_random_walk_matrix(adj: np.ndarray) -> np.ndarray:
    """Row-normalised random-walk transition matrix ``D^{-1} W`` (dense)."""
    d = adj.sum(axis=1)
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    d_mat_inv = np.diag(d_inv)
    return d_mat_inv.dot(adj).astype(np.float32)


class Model(nn.Module):
    """Adapter wrapping the upstream DCRNN architecture.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray | None
        Predefined ``(N, N)`` adjacency, injected by the runner from the
        dataset. Dual random-walk transition matrices are built from it. When
        ``None`` an identity adjacency is used as a fallback.
    input_dim : int
        Number of input channels: 1 value channel plus ``input_dim - 1``
        covariate channels.
    rnn_units : int
        Hidden size of each DCGRU cell.
    num_rnn_layers : int
        Number of stacked DCGRU layers.
    max_diffusion_step : int
        Diffusion convolution order ``K``.
    cl_decay_steps : int
        Inverse-sigmoid curriculum-learning decay (unused on CPU smoke).
    use_curriculum_learning : bool
        Whether to apply scheduled sampling during training.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        rnn_units: int = 16,
        num_rnn_layers: int = 1,
        max_diffusion_step: int = 2,
        cl_decay_steps: int = 2000,
        use_curriculum_learning: bool = False,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        if adj_mx is None:
            adj = np.eye(num_nodes, dtype=np.float32)
        else:
            adj = np.asarray(adj_mx, dtype=np.float32)
            adj = adj[:num_nodes, :num_nodes]

        # Dual random-walk transition matrices (forward + reverse).
        supports = [
            torch.from_numpy(_calculate_random_walk_matrix(adj)),
            torch.from_numpy(_calculate_random_walk_matrix(adj.T)),
        ]

        self.net = DCRNN(
            supports,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            rnn_units=rnn_units,
            num_rnn_layers=num_rnn_layers,
            max_diffusion_step=max_diffusion_step,
            cl_decay_steps=cl_decay_steps,
            use_curriculum_learning=use_curriculum_learning,
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
            Unused by DCRNN (no teacher forcing at inference).

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        # Match the configured input_dim (value + covariates).
        if history.shape[-1] >= self.input_dim:
            history = history[..., : self.input_dim]
        else:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - history.shape[-1])
            )
            history = torch.cat([history, pad], dim=-1)

        out = self.net(history, None, batch_seen=0)  # (B, pred_len, N, 1)
        return out[..., 0]
