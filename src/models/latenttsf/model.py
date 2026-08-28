"""Independent LatentTSF implementation from the latent-state paradigm."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.dlinear import DLinearBackbone


class LatentStateAutoencoder(nn.Module):
    """Per-timestep observation-to-state expansion and reconstruction."""
    def __init__(self, channels: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(channels, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, channels))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        return self.decoder(states)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(values))


def latent_alignment_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus mean cosine similarity between corresponding latent states."""
    return 1.0 - F.cosine_similarity(prediction.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()


class Model(nn.Module):
    """Two-stage autoencoder pretraining and frozen latent forecasting."""
    requires_train_target = True

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 64,
                 d_ff: int = 128, mse_weight: float = 10.0, cosine_weight: float = 15.0,
                 use_latent_norm: bool = True, kernel_size: int = 25, individual: bool = False,
                 ae_train_epochs: int = 100, ae_lr: float = 5e-4, ae_loss: str = "MAE") -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, d_ff) < 1 or min(mse_weight, cosine_weight, ae_train_epochs, ae_lr) < 0:
            raise ValueError("LatentTSF dimensions, weights, and pretraining settings are invalid")
        if ae_loss.upper() not in {"MAE", "MSE"}:
            raise ValueError("ae_loss must be MAE or MSE")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.mse_weight, self.cosine_weight = float(mse_weight), float(cosine_weight)
        self.ae_train_epochs, self.ae_lr, self.ae_loss = int(ae_train_epochs), float(ae_lr), ae_loss.upper()
        self.autoencoder = LatentStateAutoencoder(enc_in, d_model, d_ff)
        self.backbone = DLinearBackbone(c_in=d_model, seq_len=seq_len, pred_len=pred_len, kernel_size=kernel_size, individual=individual)
        self.latent_norm = nn.LayerNorm(d_model, elementwise_affine=False) if use_latent_norm else nn.Identity()
        self._autoencoder_frozen = False
        self._target: torch.Tensor | None = None
        self.train_loss_override: torch.Tensor | None = None

    def pretrain(self, train_loader, device) -> None:
        """Stage 1: learn observation reconstruction, then freeze the map."""
        if self._autoencoder_frozen:
            return
        self.autoencoder.to(device).train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)
        criterion = nn.L1Loss() if self.ae_loss == "MAE" else nn.MSELoss()
        for _ in range(self.ae_train_epochs):
            for batch in train_loader:
                values = batch[0].float().to(device)
                optimizer.zero_grad()
                criterion(self.autoencoder(values), values).backward()
                optimizer.step()
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()
        self._autoencoder_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._autoencoder_frozen:
            self.autoencoder.eval()
        return self

    def set_train_target(self, target: torch.Tensor | None) -> None:
        self._target = target

    def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f"LatentTSF expects [B, {self.seq_len}, C]")
        history_state = self.latent_norm(self.autoencoder.encode(x_enc))
        predicted_state = self.backbone(history_state)[:, -self.pred_len:]
        prediction = self.autoencoder.decode(predicted_state)
        self.train_loss_override = None
        if self._target is not None:
            target = self._target[:, -self.pred_len:].to(device=x_enc.device, dtype=x_enc.dtype)
            with torch.no_grad():
                target_state = self.autoencoder.encode(target)
            self.train_loss_override = (
                self.mse_weight * F.mse_loss(predicted_state, target_state)
                + self.cosine_weight * latent_alignment_loss(predicted_state, target_state)
            )
            self._target = None
        return prediction
