"""ModernTSF adapter for the Pathformer model."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.pathformer._upstream import Pathformer as _Pathformer


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        d_ff: int = 64,
        layer_nums: int = 3,
        k: int = 2,
        patch_size_list: list | None = None,
        num_experts_list: list | None = None,
        revin: bool = True,
        residual_connection: int = 1,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if patch_size_list is None:
            # Default: 3 layers × 4 experts each. Patch sizes must divide seq_len.
            # Use small values that work with common seq_len values (24, 48, 96).
            patch_size_list = [[8, 6, 4, 2], [6, 4, 3, 2], [4, 3, 2, 1]]
        if num_experts_list is None:
            num_experts_list = [4, 4, 4]
        self.net = _Pathformer(
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
            input_dim=1,
            d_model=d_model,
            d_ff=d_ff,
            layer_nums=layer_nums,
            k=k,
            patch_size_list=patch_size_list,
            num_experts_list=num_experts_list,
            revin=revin,
            residual_connection=residual_connection,
            batch_norm=batch_norm,
            device='cpu',
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_enc: (B, T, N)
        # Pathformer expects (B, T, N, input_dim)
        x = x_enc.unsqueeze(-1)
        result = self.net(x)
        if isinstance(result, tuple):
            out, balance_loss = result
        else:
            out = result
        # out: (B, pred_len, N, 1, 1) — squeeze trailing dims
        out = out.squeeze(-1).squeeze(-1)
        return out  # (B, pred_len, N)
