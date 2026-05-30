"""ModernTSF adapter for the STOP spatiotemporal forecasting model.

STOP (https://github.com/PoorOtterBob/STOP, under LargeST) decouples a base
``MLP`` (trend/seasonal decomposition with time-of-day and day-of-week
embeddings) from a Core_Adaptive residual-correction module. It consumes
``(B, T, N, 3)`` with channels ``[value, time_in_day, day_in_week]`` and
returns ``(B, horizon, N, 1)``.

The Core_Adaptive backcast splits ``hidden`` into ``head`` groups, so
``model_dim + 2 * prompt_dim`` must be divisible by ``head``; the defaults
below satisfy that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import to_calendar_spatiotemporal
from models.stop._upstream import MLP, STOP


class Model(nn.Module):
    """Adapter wrapping the upstream STOP model.

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
        Temporal prompt embedding dimension.
    num_layer : int
        Number of base MLP feed-forward layers.
    hid_dim : int
        Hidden dimension of the decoder network.
    tod_size : int
        Number of samples per day (time-of-day vocabulary size).
    kernel_size : int
        Series-decomposition moving-average kernel size.
    core : int
        Number of cores for the Core_Adaptive backcast (must be > 0).
    head : int
        Number of attention heads in Core_Adaptive (must divide ``hidden``).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        model_dim: int = 16,
        prompt_dim: int = 16,
        num_layer: int = 2,
        hid_dim: int = 64,
        tod_size: int = 24,
        kernel_size: int = 3,
        core: int = 4,
        head: int = 4,
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
        # STOP's MLP omits the spatial embedding: hidden = embed + tod + dow.
        hidden = model_dim + 2 * prompt_dim
        if hidden % head != 0:
            raise ValueError(
                f"STOP hidden dim ({hidden}) must be divisible by head ({head})"
            )
        model_args = {
            "extra_type": 1,
            "same": 0,
            "hid_dim": hid_dim,
            "horizon": pred_len,
        }
        self.net = STOP(
            model_args=model_args,
            stmodel=stmodel,
            dim=[hidden, hidden],
            core=core,
            ssie_dim=hidden,
            head=head,
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
            Unused by STOP.

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
