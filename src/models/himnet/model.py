"""ModernTSF adapter for the HimNet spatiotemporal graph model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/HimNet), Apache-2.0.

HimNet (KDD 2023, "Heterogeneity-Informed Meta-Parameter Learning") is a
hierarchical meta-graph GRU encoder-decoder. The spatial graph is learned
adaptively from node embeddings (and refined from a spatiotemporal embedding),
so a predefined ``adj_mx`` is *optional*; when supplied it is used to warm-start
the node embeddings. The upstream ``forward`` is
``forward(x, y_cov, labels, batches_seen)`` with:

* ``x``      : ``(B, T, N, input_dim)`` history, channel 0 = value,
               channel 1 = time-of-day in ``[0, 1)``,
               channel 2 = day-of-week as an *integer index* in ``[0, 7)``.
* ``y_cov``  : ``(B, out_steps, N, ycov_dim)`` future calendar covariates
               ``[time_of_day, day_of_week]``.

This adapter rebuilds those tensors from ModernTSF's
``(x_enc, x_mark_enc, x_dec, x_mark_dec)`` 4-tuple via
``models._external.marks.to_spatiotemporal`` and returns ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.himnet._upstream import HimNet


class Model(nn.Module):
    """Adapter wrapping the upstream HimNet model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray | None
        Optional ``(N, N)`` predefined adjacency injected by the runner. HimNet
        learns its graph adaptively; when an adjacency is supplied it is used to
        warm-start the node embeddings (registered as a buffer for device
        portability) but the model still adapts it during training.
    input_dim : int
        Number of input channels fed to HimNet (value + calendar). Default 3.
    hidden_dim, num_layers, cheb_k, node_embedding_dim, st_embedding_dim,
    tod_embedding_dim, dow_embedding_dim, steps_per_day, use_teacher_forcing
        HimNet hyper-parameters.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx: np.ndarray | None = None,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
        cheb_k: int = 2,
        node_embedding_dim: int = 8,
        st_embedding_dim: int = 8,
        tod_embedding_dim: int = 8,
        dow_embedding_dim: int = 8,
        steps_per_day: int = 288,
        use_teacher_forcing: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.ycov_dim = 2  # [time_of_day, day_of_week]

        self.net = HimNet(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            out_steps=pred_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cheb_k=cheb_k,
            ycov_dim=self.ycov_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            node_embedding_dim=node_embedding_dim,
            st_embedding_dim=st_embedding_dim,
            use_teacher_forcing=use_teacher_forcing,
            steps_per_day=steps_per_day,
        )

        # HimNet's graph is adaptive, but if a predefined adjacency is injected
        # we keep it as a buffer (device-portable) and use it to warm-start the
        # learned node embeddings, mixing structural priors into the model.
        if adj_mx is not None:
            adj = np.asarray(adj_mx, dtype=np.float32)
            self.register_buffer("adj_mx", torch.from_numpy(adj))
            self._warm_start_node_embedding(adj)
        else:
            self.adj_mx = None

    def _warm_start_node_embedding(self, adj: np.ndarray) -> None:
        """Initialise node embeddings from the predefined adjacency.

        Uses the top-``node_embedding_dim`` symmetric eigenvectors of the
        adjacency as a structural prior. Purely a better starting point — the
        embeddings remain trainable, so the graph stays adaptive.
        """
        try:
            sym = (adj + adj.T) / 2.0
            w, v = np.linalg.eigh(sym)
            k = self.net.node_embedding.shape[1]
            order = np.argsort(-w)[:k]
            emb = v[:, order].astype(np.float32)  # (N, k)
            if emb.shape[1] < k:
                pad = np.zeros((emb.shape[0], k - emb.shape[1]), dtype=np.float32)
                emb = np.concatenate([emb, pad], axis=1)
            with torch.no_grad():
                self.net.node_embedding.copy_(torch.from_numpy(emb))
        except np.linalg.LinAlgError:
            pass

    def _calendar(self, marks: torch.Tensor | None, length: int, b: int) -> torch.Tensor:
        """Build ``(B, length, N, 2)`` = ``[time_of_day, day_of_week_index]``.

        ``to_spatiotemporal`` yields ``[value, time_of_day, day_of_week/7]``;
        HimNet indexes ``dow_embedding`` by an *integer* day-of-week, so the
        normalised day-of-week channel is scaled back to ``[0, 7)``.
        """
        dummy_value = marks.new_zeros((b, length, self.num_nodes)) if marks is not None \
            else None
        if dummy_value is None:
            # No marks at all: zeros for tod/dow.
            return torch.zeros(
                (b, length, self.num_nodes, 2),
            )
        st = to_spatiotemporal(dummy_value, marks)  # (B, length, N, 1 + F)
        cov = st[..., 1:3]  # (B, length, N, 2) -> [tod, dow/7]
        if cov.shape[-1] < 2:
            pad = cov.new_zeros((*cov.shape[:-1], 2 - cov.shape[-1]))
            cov = torch.cat([cov, pad], dim=-1)
        tod = cov[..., 0:1]
        dow = (cov[..., 1:2] * 7.0).round().clamp(0, 6)  # integer index 0..6
        return torch.cat([tod, dow], dim=-1)

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
            Input covariate marks ``(B, seq_len, N, F)`` (spatiotemporal) or raw
            calendar stamps ``(B, seq_len, 6)``.
        x_dec : torch.Tensor, optional
            Unused (decoder is autoregressive on the value channel).
        x_mark_dec : torch.Tensor, optional
            Future covariate marks ``(B, pred_len, N, F)`` or raw stamps.
        mask
            Unused.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        b = x_enc.shape[0]
        device = x_enc.device

        # History: [value, time_of_day, day_of_week_index].
        history_full = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1 + F)
        value = history_full[..., 0:1]
        in_cal = self._calendar(x_mark_enc, self.seq_len, b).to(device)
        history = torch.cat([value, in_cal], dim=-1)  # (B, T, N, 3)

        # Future calendar covariates for the decoder.
        y_cov = self._calendar(x_mark_dec, self.pred_len, b).to(device)

        out = self.net(history, y_cov, labels=None, batches_seen=0)
        # out: (B, pred_len, N, output_dim)
        return out[..., 0]
