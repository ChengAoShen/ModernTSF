"""ModernTSF adapter for the GWNet (Graph WaveNet) spatiotemporal model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/GWNet), Apache-2.0.

Graph WaveNet (IJCAI 2019) combines a stack of dilated temporal convolutions
with a graph-convolution module that mixes a *predefined* adjacency (the
injected ``adj_mx``) with an *adaptive*, learned adjacency. ModernTSF feeds the
model a value tensor ``(B, seq_len, N)`` plus node-structured covariate marks
``(B, seq_len, N, F)``; this adapter reassembles the BasicTS spatiotemporal
layout ``(B, L, N, 1 + F)`` (value channel 0, then the first
``input_dim - 1`` calendar covariates ``[time_in_day, day_in_week]``), drives
the upstream module with the BasicTS forward signature, and squeezes the output
channel back to ``(B, pred_len, N)``.

The injected ``(N, N)`` ``adj_mx`` is passed as one entry of the ``supports``
list while the adaptive adjacency is kept on (``addaptadj=True``). The predefined
adjacency is stored as a registered buffer so it follows the model's device; no
tensor is created on a hardcoded CUDA device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.gwnet._upstream import GraphWaveNet


class Model(nn.Module):
    """Adapter wrapping the upstream Graph WaveNet architecture.

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
        dataset. When ``None`` only the adaptive adjacency is used.
    input_dim : int
        Number of input channels fed to the network: 1 value channel plus
        ``input_dim - 1`` calendar covariates ``[time_in_day, day_in_week]``.
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
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        # Predefined adjacency as a buffer so it follows the model device.
        supports = None
        if adj_mx is not None:
            adj = np.asarray(adj_mx, dtype=np.float32)
            self.register_buffer("adj_support", torch.from_numpy(adj))
            supports = [self.adj_support]

        self.net = GraphWaveNet(
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=True,
            addaptadj=True,
            aptinit=None,
            in_dim=input_dim,
            out_dim=pred_len,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
        )
        # The upstream constructor may have replaced ``supports`` with ``[]``
        # (when adj_mx is None); keep our buffer reference list in sync.
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
            calendar stamps ``(B, seq_len, 6)``.
        x_dec, x_mark_dec, mask
            Unused by Graph WaveNet.

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
            out = out[..., 0]
        return out
