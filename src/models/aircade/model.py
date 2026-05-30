"""ModernTSF adapter for the AirCade air-quality forecasting model.

AirCade (https://github.com/PoorOtterBob/AirCade) forecasts a target series
from its history plus future covariates. It consumes:

* ``x`` : ``(B, time_step, N, input_dim)`` history; channel 0 is the value,
          channels ``1:`` are covariates.
* ``y`` : ``(B, time_step, N, input_dim - 1)`` future covariates.

and returns ``(B, time_step, N, output_dim)``. The temporal length is fixed
at ``time_step`` for input, future and output, so this adapter sets
``time_step = seq_len`` and requires ``pred_len == seq_len``.

The upstream ``DK_MSA`` module hard-codes a 184-node adaptive embedding
(tied to the original dataset). To keep ``_upstream.py`` verbatim we resize
those parameters in place after construction for the configured node count.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import (
    TIME_FEATURES,
    coerce_time_length,
    future_time_features,
    to_spatiotemporal,
)
from models.aircade._upstream import DK_MSA, AirCade


class Model(nn.Module):
    """Adapter wrapping the upstream AirCade model.

    Parameters
    ----------
    seq_len : int
        Input sequence length (also the AirCade ``time_step``).
    pred_len : int
        Forecast horizon (must equal ``seq_len``).
    enc_in : int
        Number of spatial nodes (channels).
    input_embedding_dim : int
        Input projection embedding dimension.
    adaptive_embedding_dim : int
        DK-Prompt adaptive embedding dimension.
    feed_forward_dim : int
        Feed-forward hidden dimension.
    num_heads : int
        Number of attention heads (must divide the model dimension).
    num_layers : int
        Number of gated / inverse transformer layers.
    node_embed_dim : int
        Inner dimension of the resized DK_MSA adaptive node embedding.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int | None = None,
        input_embedding_dim: int = 16,
        adaptive_embedding_dim: int = 24,
        feed_forward_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        node_embed_dim: int = 10,
    ) -> None:
        super().__init__()
        if pred_len != seq_len:
            raise ValueError(
                f"AirCade requires pred_len == seq_len (got pred_len={pred_len}, "
                f"seq_len={seq_len}); its temporal length is fixed at time_step."
            )
        self.seq_len = seq_len
        input_dim = 1 + (TIME_FEATURES if cov_dim is None else cov_dim)
        self.net = AirCade(
            time_step=seq_len,
            input_embedding_dim=input_embedding_dim,
            DK_Prompt_adaptive_embedding=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_mixed=True,
            num_nodes=enc_in,
            input_dim=input_dim,
            output_dim=1,
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
        )
        self._resize_node_embeddings(enc_in, node_embed_dim)

    def _resize_node_embeddings(self, num_nodes: int, embed_dim: int) -> None:
        """Resize every DK_MSA adaptive node embedding to ``num_nodes``.

        The upstream module fixes these at ``(184, 10)`` / ``(10, 184)``; we
        re-initialize them to ``(num_nodes, embed_dim)`` / ``(embed_dim,
        num_nodes)`` so the model works for an arbitrary node count.
        """
        for module in self.net.modules():
            if isinstance(module, DK_MSA):
                emb1 = torch.empty(num_nodes, embed_dim)
                emb2 = torch.empty(embed_dim, num_nodes)
                nn.init.xavier_uniform_(emb1)
                nn.init.xavier_uniform_(emb2)
                module.node_emb1 = nn.Parameter(emb1)
                module.node_emb2 = nn.Parameter(emb2)

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
            Raw input marks of shape ``(B, seq_len, 6)``.
        x_dec : torch.Tensor, optional
            Unused (AirCade uses future covariates, not future values).
        x_mark_dec : torch.Tensor, optional
            Raw future marks of shape ``(B, label_len + pred_len, 6)``.
        mask : torch.Tensor, optional
            Unused.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        b, t, n = x_enc.shape
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((b, t, 6))
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1 + F)

        future_marks = x_mark_enc if x_mark_dec is None else x_mark_dec
        future_marks = coerce_time_length(future_marks, self.seq_len)
        future = future_time_features(future_marks, n)  # (B, seq_len, N, F)

        out = self.net(history, future)  # (B, time_step, N, 1)
        return out.squeeze(-1)
