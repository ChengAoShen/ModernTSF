"""ModernTSF adapter for the MTGNN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/MTGNN), Apache-2.0.

MTGNN (Wu et al., SIGKDD 2020) learns the graph structure via a
graph-learning layer, and can optionally fuse a predefined adjacency. It
consumes a spatiotemporal tensor of shape ``(B, L, N, C)`` (channel 0 is the
value, the rest are covariates) and returns ``(B, pred_len, N, 1)``.

This adapter converts ModernTSF's ``(x_enc, x_mark_enc)`` into the
``(B, L, N, 1 + F)`` layout via :func:`to_spatiotemporal`, drives the upstream
module with the BasicTS forward signature, and squeezes the output channel
back to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.mtgnn._upstream import MTGNN


class Model(nn.Module):
    """Adapter wrapping the upstream MTGNN model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes (``N``).
    adj_mx : np.ndarray | None
        Optional predefined ``(N, N)`` adjacency. When supplied and
        ``build_adj`` is False, MTGNN uses it directly; otherwise the
        adaptive graph-learning layer builds the graph (predefined adj
        ignored). Injected by the runner from the dataset.
    input_dim : int
        Number of input channels per node (value + covariates).
    gcn_depth : int
        Mix-hop propagation depth.
    subgraph_size : int
        Top-k neighbours kept per node in the learned graph.
    node_dim : int
        Node-embedding dimension for graph learning.
    conv_channels, residual_channels, skip_channels, end_channels : int
        Channel widths of the temporal-convolution stack.
    layers : int
        Number of spatiotemporal layers.
    dropout : float
        Dropout rate.
    propalpha : float
        Mix-hop retention ratio.
    tanhalpha : float
        Graph-learning saturation factor.
    dilation_exponential : int
        Dilation growth factor.
    build_adj : bool
        Learn the graph adaptively (True) or use ``adj_mx`` directly (False).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        gcn_depth: int = 2,
        subgraph_size: int = 20,
        node_dim: int = 40,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        layers: int = 3,
        dropout: float = 0.3,
        propalpha: float = 0.05,
        tanhalpha: float = 3.0,
        dilation_exponential: int = 1,
        build_adj: bool = True,
    ) -> None:
        super().__init__()
        # subgraph_size cannot exceed the node count.
        subgraph_size = min(subgraph_size, num_nodes)

        predefined_A = None
        buildA_true = bool(build_adj)
        if adj_mx is not None:
            predefined_A = torch.from_numpy(np.asarray(adj_mx, dtype=np.float32))
            if not buildA_true:
                # Use the predefined adjacency directly.
                pass
        else:
            # No predefined adjacency: must learn it adaptively.
            buildA_true = True

        self.net = MTGNN(
            gcn_true=True,
            buildA_true=buildA_true,
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            predefined_A=predefined_A,
            static_feat=None,
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            dilation_exponential=dilation_exponential,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            seq_length=seq_len,
            in_dim=input_dim,
            out_dim=pred_len,
            layers=layers,
            propalpha=propalpha,
            tanhalpha=tanhalpha,
            layer_norm_affline=True,
        )
        self.input_dim = input_dim

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
            Marks: raw ``(B, seq_len, 6)`` or node covariates
            ``(B, seq_len, N, F)``.
        x_dec, x_mark_dec, mask
            Unused by MTGNN.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1 + F)
        c = history.shape[-1]
        if c < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - c)
            )
            history = torch.cat([history, pad], dim=-1)
        elif c > self.input_dim:
            history = history[..., : self.input_dim]

        out = self.net(
            history, None, batch_seen=0, epoch=0, train=self.training
        )  # (B, pred_len, N, 1)
        if out.dim() == 4:
            return out[..., 0]
        return out
