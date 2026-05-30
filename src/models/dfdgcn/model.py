"""ModernTSF adapter for the DFDGCN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/DFDGCN), Apache-2.0. The architecture itself matches the official
DFDGCN reference implementation released by the BasicTS authors
(GestaltCogTeam/DFDGCN, MIT) and is vendored locally in ``_upstream.py``.

DFDGCN ("Dynamic Frequency Domain Graph Convolution Network") builds on Graph
WaveNet: dilated temporal convolutions plus graph convolution mixing a
*predefined* adjacency, a learned *adaptive* adjacency, and a per-batch
*dynamic frequency-domain* graph derived from the FFT magnitude spectrum of the
traffic series together with node-identity and time-of-day / day-of-week
embeddings.

ModernTSF feeds the model a value tensor ``(B, seq_len, N)`` plus
node-structured covariate marks ``(B, seq_len, N, F)`` (or raw calendar
stamps). This adapter reassembles the BasicTS spatiotemporal layout
``(B, L, N, 1 + F)`` (value channel 0, then ``[time_in_day, day_in_week]``),
drives the upstream module with the BasicTS forward signature, and squeezes the
output channel back to ``(B, pred_len, N)``.

The injected ``(N, N)`` ``adj_mx`` is converted to the DCRNN / Graph-WaveNet
"double transition" supports ``[P, P^T]`` (``P = D^{-1} A``) and stored as
registered buffers so they follow the model device; no tensor is created on a
hardcoded CUDA device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.dfdgcn._upstream import DFDGCN


def _transition_matrix(adj: np.ndarray) -> np.ndarray:
    """Row-normalized transition matrix ``P = D^{-1} A`` (DCRNN / GWNet)."""
    adj = np.asarray(adj, dtype=np.float32)
    row_sum = adj.sum(axis=1)
    d_inv = np.power(row_sum, -1, where=row_sum != 0)
    d_inv[np.isinf(d_inv)] = 0.0
    d_inv[row_sum == 0] = 0.0
    return (np.diag(d_inv) @ adj).astype(np.float32)


class Model(nn.Module):
    """Adapter wrapping the upstream DFDGCN architecture.

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
        dataset. Converted to double-transition supports ``[P, P^T]``. When
        ``None`` only the adaptive / dynamic graphs are used.
    input_dim : int
        Number of channels in the assembled history tensor (value + calendar
        covariates). DFDGCN consumes channels ``0:2`` for the TCN and channels
        ``1`` / ``2`` as time-of-day / day-of-week embedding indices.
    dropout : float
        Dropout rate inside the graph-conv blocks.
    residual_channels, dilation_channels : int
        Channel widths of the dilated TCN backbone.
    skip_channels, end_channels : int
        Channel widths of the skip / output 1x1 convolutions.
    kernel_size : int
        Temporal convolution kernel size.
    blocks, layers : int
        Number of WaveNet blocks and dilated layers per block.
    a : float
        Scaling factor applied to the normalized FFT magnitude spectrum.
    fft_emb, identity_emb, hidden_emb : int
        Embedding dimensions of the dynamic frequency-domain graph builder.
    subgraph : int
        Top-k neighbours kept per node in the dynamic graph mask.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        dropout: float = 0.3,
        residual_channels: int = 16,
        dilation_channels: int = 16,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        blocks: int = 2,
        layers: int = 2,
        a: float = 1.0,
        fft_emb: int = 10,
        identity_emb: int = 10,
        hidden_emb: int = 30,
        subgraph: int = 20,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = max(input_dim, 3)

        # Double-transition supports as buffers so they follow the device.
        supports = None
        if adj_mx is not None:
            adj = np.asarray(adj_mx, dtype=np.float32)
            p_fwd = _transition_matrix(adj)
            p_bwd = _transition_matrix(adj.T)
            self.register_buffer("support_fwd", torch.from_numpy(p_fwd))
            self.register_buffer("support_bwd", torch.from_numpy(p_bwd))
            supports = [self.support_fwd, self.support_bwd]

        # The dynamic graph never keeps more neighbours than there are nodes.
        subgraph = min(subgraph, num_nodes)

        self.net = DFDGCN(
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=2,
            out_dim=pred_len,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
            a=a,
            seq_len=seq_len,
            affine=False,
            fft_emb=fft_emb,
            identity_emb=identity_emb,
            hidden_emb=hidden_emb,
            subgraph=subgraph,
        )
        # Keep the buffer references in the live supports list.
        if supports is not None:
            self.net.supports = supports

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
            calendar stamps ``(B, seq_len, 6)``. Channels 1/2 carry
            ``[time_in_day, day_in_week]``.
        x_dec, x_mark_dec, mask
            Unused by DFDGCN.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        # Ensure at least [value, time_in_day, day_in_week].
        if history.shape[-1] < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - history.shape[-1])
            )
            history = torch.cat([history, pad], dim=-1)
        history = history[..., : self.input_dim]

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
