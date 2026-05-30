"""ModernTSF adapter for the STDN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STDN), Apache-2.0.

STDN (Spatial-Temporal Decomposition / Dynamic Network) consumes a value
tensor ``X`` of shape ``(B, L, N, 1)``, an integer time-encoding ``TE`` of
shape ``(B, L + pred_len, 2)`` ordered ``[day_of_week, time_of_day]``, and a
Laplacian positional encoding ``lpls`` of shape ``(N, 32)`` derived from a
predefined ``(N, N)`` adjacency. It returns ``(B, pred_len, N, 1)``.

This adapter converts ModernTSF's ``(x_enc, marks)`` into that layout. The
node-structured covariate marks (``cauair_st`` bundle) carry the normalized
calendar features ``[time_in_day, day_in_week]`` in channels ``1`` and ``2`` of
the spatiotemporal tensor; these are scaled to integer indices for ``TE``. The
predefined adjacency (injected by the runner as ``adj_mx``) is converted to a
Laplacian positional encoding and registered as a buffer on first forward.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.stdn._upstream import STDN

# SEmbedding hardcodes an input Linear(32, 32); the Laplacian PE must therefore
# have exactly this many feature columns (pad/truncate to fit).
_LAPE_DIM = 32


def _calculate_normalized_laplacian(adj: np.ndarray):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = int(np.sum(np.where(d, 0, 1)))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian, isolated_point_num


def _cal_lape(adj_mx: np.ndarray, lape_dim: int = _LAPE_DIM) -> np.ndarray:
    """Laplacian positional encoding, padded/truncated to ``lape_dim`` columns."""
    L, isolated_point_num = _calculate_normalized_laplacian(adj_mx)
    eig_val, eig_vec = np.linalg.eig(L.toarray())
    idx = eig_val.argsort()
    eig_val, eig_vec = eig_val[idx], np.real(eig_vec[:, idx])
    start = isolated_point_num + 1
    laplacian_pe = eig_vec[:, start: lape_dim + start]
    n = adj_mx.shape[0]
    pe = np.zeros((n, lape_dim), dtype=np.float32)
    cols = min(lape_dim, laplacian_pe.shape[1])
    pe[:, :cols] = laplacian_pe[:, :cols]
    return pe


class Model(nn.Module):
    """Adapter wrapping the upstream STDN model.

    Parameters
    ----------
    seq_len : int
        Input (history) sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray, optional
        Predefined ``(N, N)`` adjacency injected by the runner. Used to build
        the Laplacian positional encoding. A self-loop ring is synthesised if
        absent.
    input_dim : int
        Number of input channels in the source bundle (unused by STDN beyond
        the value channel; kept for interface parity).
    time_slice_size : int
        Minutes per time slot (``1440 / time_slice_size`` slots per day). For
        the hourly smoke bundle this is ``60`` (24 slots/day).
    K : int
        Number of attention heads.
    d : int
        Dimension per attention head (model dim ``D = K * d``).
    L : int
        Number of attention decoder layers.
    order : int
        Diffusion order of the dynamic GCN.
    reference : int
        Inducing-set size for the attention decoder.
    out_channels : int
        Output channels per node (``1`` for univariate forecasting).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        time_slice_size: int = 60,
        K: int = 4,
        d: int = 8,
        L: int = 1,
        order: int = 2,
        reference: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.time_slice_size = time_slice_size
        self.slots_per_day = int(1440 / time_slice_size)

        args = {
            "Data": {
                "num_of_vertices": num_nodes,
                "time_slice_size": time_slice_size,
                "dataset_name": "modern_tsf",
            },
            "Training": {
                "L": L,
                "K": K,
                "d": d,
                "node_miss_rate": 0.0,
                "T_miss_len": 0,
                "order": order,
                "reference": reference,
                "num_his": seq_len,
                "num_pred": pred_len,
                "in_channels": 1,
                "out_channels": out_channels,
            },
        }
        self.net = STDN(args, bn_decay=0.1)

        # Build the Laplacian positional encoding from the predefined adjacency.
        if adj_mx is None:
            adj = np.eye(num_nodes, dtype=np.float32)
            for i in range(num_nodes):
                adj[i, (i + 1) % num_nodes] = 1.0
                adj[i, (i - 1) % num_nodes] = 1.0
        else:
            adj = np.asarray(adj_mx, dtype=np.float64)
            adj = adj + np.eye(adj.shape[0])
        lpls = _cal_lape(adj)
        self.register_buffer("lpls", torch.from_numpy(lpls).float(), persistent=False)

    def _build_te(
        self,
        x_mark_enc: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        st_hist: torch.Tensor,
    ) -> torch.Tensor:
        """Build the integer time-encoding ``TE`` of shape ``(B, L + pred, 2)``.

        Columns are ``[day_of_week_index, time_of_day_index]``. The normalized
        calendar features ``[time_in_day, day_in_week]`` live in channels 1 and
        2 of the spatiotemporal tensor (broadcast identically across nodes), so
        we take node 0 and scale them to integer indices.
        """
        device = st_hist.device
        b = st_hist.shape[0]

        def from_st(st: torch.Tensor) -> torch.Tensor:
            # st: (B, T, N, 1 + F). Channel 1 = time_in_day, 2 = day_in_week.
            if st.shape[-1] >= 3:
                tid = st[:, :, 0, 1]  # (B, T) normalized [0, 1)
                dow = st[:, :, 0, 2]  # (B, T) normalized [0, 1)
            else:
                tid = st.new_zeros((st.shape[0], st.shape[1]))
                dow = st.new_zeros((st.shape[0], st.shape[1]))
            dow_idx = (dow * 7.0).round()
            tid_idx = (tid * self.slots_per_day).round()
            return torch.stack([dow_idx, tid_idx], dim=-1)  # (B, T, 2)

        te_hist = from_st(st_hist)  # (B, seq_len, 2)

        if x_mark_dec is not None:
            st_fut = to_spatiotemporal(
                torch.zeros(
                    b, x_mark_dec.shape[1], self.num_nodes, device=device
                ),
                x_mark_dec,
            )
            te_fut = from_st(st_fut)[:, -self.pred_len:]  # (B, pred_len, 2)
        else:
            # No future marks: continue the day index, hold day-of-week.
            last = te_hist[:, -1:]  # (B, 1, 2)
            steps = torch.arange(1, self.pred_len + 1, device=device).view(1, -1, 1)
            tid = (last[..., 1:2] + steps) % self.slots_per_day
            dow = last[..., 0:1].expand(b, self.pred_len, 1)
            te_fut = torch.cat([dow, tid], dim=-1)

        te = torch.cat([te_hist, te_fut], dim=1)  # (B, L + pred, 2)
        return te.to(device)

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
            Marks ``(B, seq_len, N, F)`` (node covariates) or ``(B, seq_len, 6)``.
        x_dec, mask
            Unused.
        x_mark_dec : torch.Tensor, optional
            Future marks, used to build the future time-encoding block.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        st_hist = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        value = st_hist[..., :1]  # (B, L, N, 1)
        te = self._build_te(x_mark_enc, x_mark_dec, st_hist)  # (B, L + pred, 2)

        out = self.net(value, te, self.lpls, "train" if self.training else "test")
        # out: (B, pred_len, N, out_channels)
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
