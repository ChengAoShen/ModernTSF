"""ModernTSF adapter for the STGODE spatiotemporal graph-ODE model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STGODE), Apache-2.0.

STGODE (Spatial-Temporal Graph ODE Networks, KDD 2021) stacks dilated temporal
convolutions with a neural-ODE graph-convolution block over **two** predefined
graphs: a *spatial* adjacency and a *semantic* (DTW-based) adjacency. ModernTSF
feeds the model a value tensor ``(B, seq_len, N)`` plus node-structured
covariate marks ``(B, seq_len, N, F)``; this adapter reassembles the BasicTS
spatiotemporal layout ``(B, L, N, 1 + F)`` (value channel 0, then the first
``input_dim - 1`` calendar covariates ``[time_in_day, day_in_week]``), drives
the upstream module with the BasicTS forward signature, and squeezes the output
channel back to ``(B, pred_len, N)``.

The injected ``(N, N)`` ``adj_mx`` is degree-normalized with the upstream
``get_normalized_adj`` to build the spatial graph ``A_sp_hat``. Upstream derives
the semantic graph from a DTW distance over the full training series plus the
``fastdtw`` dependency; to stay dependency-free we derive the semantic graph
from the same injected adjacency (also degree-normalized). Both matrices are
stored as registered buffers so they follow the model's device; no tensor is
created on a hardcoded CUDA device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.stgode._upstream import ODEGCN


def get_normalized_adj(A: np.ndarray) -> np.ndarray:
    """Degree-normalized adjacency, matching the upstream STGODE recipe.

    Returns ``alpha/2 * (I + D^{-1/2} A D^{-1/2})`` with ``alpha = 0.8``.
    """
    alpha = 0.8
    A = np.asarray(A, dtype=np.float32)
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(
        np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1))
    )
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return A_reg.astype(np.float32)


class Model(nn.Module):
    """Adapter wrapping the upstream STGODE (ODEGCN) architecture.

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
        dataset. When ``None`` an identity graph is used.
    input_dim : int
        Number of input channels fed to the network: 1 value channel plus
        ``input_dim - 1`` calendar covariates ``[time_in_day, day_in_week]``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        if adj_mx is None:
            adj = np.eye(num_nodes, dtype=np.float32)
        else:
            adj = np.asarray(adj_mx, dtype=np.float32)

        # Spatial graph from the injected adjacency. The semantic graph is
        # derived from the same adjacency (DTW + fastdtw avoided), keeping the
        # dual-graph structure intact and dependency-free.
        a_sp = get_normalized_adj(adj)
        a_se = get_normalized_adj(adj)
        self.register_buffer("a_sp_hat", torch.from_numpy(a_sp))
        self.register_buffer("a_se_hat", torch.from_numpy(a_se))

        self.net = ODEGCN(
            num_nodes=num_nodes,
            num_features=input_dim,
            num_timesteps_input=seq_len,
            num_timesteps_output=pred_len,
            A_sp_hat=self.a_sp_hat,
            A_se_hat=self.a_se_hat,
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
            Unused by STGODE.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        # Keep value channel 0 + the calendar covariates the network expects.
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
            out = out[..., 0]  # (B, pred_len, N)
        return out
