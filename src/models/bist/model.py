"""ModernTSF adapter for the BiST spatiotemporal forecasting model.

BiST (https://github.com/PoorOtterBob/BiST) consumes a spatiotemporal tensor
of shape ``(B, T, N, 1 + F)`` where channel 0 is the value and the remaining
channels are normalized calendar features ``[time_in_day, day_in_week]``. It
returns ``(B, horizon, N, 1)``.

This adapter converts ModernTSF's ``(x_enc, marks)`` into that layout, drives
the upstream model (a decoupled base ``MLP`` plus residual propagation), and
squeezes the output channel back to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import to_calendar_spatiotemporal
from models.bist._upstream import MLP, BiST


class Model(nn.Module):
    """Adapter wrapping the upstream BiST model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    enc_in : int
        Number of spatial nodes (channels).
    model_dim : int
        Base MLP embedding dimension.
    prompt_dim : int
        Temporal / spatial prompt embedding dimension.
    num_layer : int
        Number of base MLP feed-forward layers.
    hid_dim : int
        Hidden dimension of the backcast / decoder networks.
    tod_size : int
        Number of samples per day (time-of-day vocabulary size).
    kernel_size : int
        Series-decomposition moving-average kernel size.
    rp_layer : int
        Number of residual-propagation layers.
    adaptive_adj_dim : int
        Node-embedding dimension for the learned adaptive graph.
    core : int
        Number of cores for the optional Core_Adaptive backcast (0 disables).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        model_dim: int = 32,
        prompt_dim: int = 32,
        num_layer: int = 2,
        hid_dim: int = 64,
        tod_size: int = 24,
        kernel_size: int = 3,
        rp_layer: int = 1,
        adaptive_adj_dim: int = 10,
        core: int = 0,
    ) -> None:
        super().__init__()
        base_args = dict(
            node_num=enc_in,
            input_dim=3,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
        )
        stmodel = MLP(
            num_layer=num_layer,
            model_dim=model_dim,
            prompt_dim=prompt_dim,
            tod_size=tod_size,
            kernel_size=kernel_size,
            **base_args,
        )
        hidden = model_dim + 3 * prompt_dim
        model_args = {
            "extra_type": 1,
            "same": 0,
            "hid_dim": hid_dim,
            "horizon": pred_len,
            "predefined_adj": 0,
            "adjs": [],
            "rp_layer": rp_layer,
            "datadriven_adj": 0,
            "datadriven_adj_dim": 32,
            "datadriven_adj_head": 0,
            "adaptive_adj": 1,
            "adaptive_adj_dim": adaptive_adj_dim,
            "mrf": 1,
            "use_global_opt": False,
        }
        self.net = BiST(
            model_args=model_args,
            stmodel=stmodel,
            dim=[hidden, hidden],
            core=core,
            **base_args,
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
            Raw input marks of shape ``(B, seq_len, 6)``.
        x_dec, x_mark_dec, mask
            Unused by BiST.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_calendar_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 3)
        out = self.net(st_input)  # (B, horizon, N, 1)
        return out.squeeze(-1)
