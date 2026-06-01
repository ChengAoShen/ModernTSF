"""ModernTSF adapter for LightGBMTS.

This is a PyTorch-native time-series forecasting adapter for the LightGBMTS
classical/ML baseline family. It follows the ModernTSF ``nn.Module`` interface
and can run on CPU, CUDA, or MPS through the standard trainer.
"""

from __future__ import annotations

import torch.nn as nn

from models._ml_tsf import MLTSFModel


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        num_layers: int = 1,
        num_estimators: int = 16,
        tree_depth: int = 3,
        num_prototypes: int = 32,
        kernel_gamma: float = 0.1,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.model = MLTSFModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            family="lightgbm",
            variant="LightGBMTS",
            d_model=d_model,
            dropout=dropout,
            num_layers=num_layers,
            num_estimators=num_estimators,
            tree_depth=tree_depth,
            num_prototypes=num_prototypes,
            kernel_gamma=kernel_gamma,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            use_revin=use_revin,
        )

    @property
    def aux_loss(self):
        return self.model.aux_loss

    def forward(self, x, *args):
        return self.model(x)
