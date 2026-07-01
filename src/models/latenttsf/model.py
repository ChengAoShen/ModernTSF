"""ModernTSF faithful adapter for LatentTSF (ICML 2026).

*From Observations to States: Latent Time Series Forecasting.*

Two-stage latent forecasting, reproducing the upstream recipe
(https://github.com/Muyiiiii/LatentTSF):

- **Stage 1 (pretrain).** A per-timestep MLP autoencoder ``(E, D)`` is trained to
  reconstruct observations, then **frozen**. ``E`` maps each timestep's
  ``enc_in``-dim observation to a ``d_model``-dim latent state; ``D`` maps it
  back. The window length is irrelevant to the AE (it acts per timestep), so the
  AE pretrained on the forecasting windows transfers directly.
- **Stage 2 (forecast).** A backbone forecaster ``f`` is trained **entirely in
  the frozen latent space**::

      X -E-> Z_X -f-> Ẑ_Y -D-> Ŷ ,        Z_Y = E(Y)

  The training objective (paper Eq. 5) lives only in latent space::

      L = mse_weight * ||Z_Y - Ẑ_Y||_F^2  +  cosine_weight * (1 - cos(Z_Y, Ẑ_Y))

  The observation-space MSE is **not** part of the default objective — the
  paper's optional perceptual loss is off by default (Sec. 5.3.1). Defaults
  ``mse_weight=10`` (α), ``cosine_weight=15`` (β) follow Sec. 5.3.2.

Integration with ModernTSF's trainer uses three opt-in, no-op-for-other-models
conventions (see ``benchmark.runner.trainer`` / ``benchmark.runner.run_one``):

- ``pretrain(train_loader, device)`` — ``run_one`` calls it once before training
  to pretrain + freeze the AE (Stage 1).
- ``requires_train_target = True`` + ``set_train_target(y_or_none)`` — the trainer feeds the raw future
  target each training step so the model can encode ``Z_Y = E(Y)`` for the
  latent loss.
- ``train_loss_override`` — when set during a forward, the trainer uses it as the
  training loss (replacing the observation-space criterion). Validation /
  early-stopping still use the configured criterion (observation MSE), matching
  the upstream ``valid_epoch`` signal.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dlinear.model import Model as DLinearBackbone


class _MLPAutoEncoder(nn.Module):
    """Per-timestep MLP autoencoder on the feature dimension.

    Input/output: ``(B, T, enc_in)``; latent: ``(B, T, d_model)``. Each
    timestep's observation vector is encoded independently, matching the
    upstream ``AutoEncoder`` in ``my_AE.py``.
    """

    def __init__(self, enc_in: int, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, enc_in),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _cosine_align_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """``1 - mean(cos)`` over per-token latent vectors (upstream definition)."""
    p = z_pred.reshape(-1, z_pred.shape[-1])
    t = z_target.reshape(-1, z_target.shape[-1])
    return 1.0 - F.cosine_similarity(p, t, dim=-1).mean()


class Model(nn.Module):
    """LatentTSF: forecast in the latent space of a frozen pretrained AE.

    Parameters
    ----------
    seq_len, pred_len, enc_in : int
        Forecasting task dimensions (input length, horizon, channels).
    d_model, d_ff : int
        Latent state dimension and AE hidden width.
    mse_weight, cosine_weight : float
        Latent prediction (α) and alignment (β) loss weights. Paper defaults
        ``10`` and ``15``.
    use_latent_norm : bool
        Apply a fixed (non-affine) LayerNorm to the encoder input ``Z_X`` (and
        therefore to the forecaster input). The latent target ``Z_Y`` is left
        un-normalized, matching upstream.
    kernel_size, individual : int, bool
        DLinear backbone hyper-parameters (the paper's primary Table-1 backbone).
    ae_train_epochs, ae_lr, ae_loss : int, float, str
        Stage-1 AE pretraining schedule (used only when ``autoencoder_path`` is
        empty). ``ae_loss`` is ``"MAE"`` (L1, the shipped-checkpoint recipe) or
        ``"MSE"``. Set ``ae_train_epochs=0`` to skip pretraining (random frozen
        AE — for debugging only).
    autoencoder_path : str
        Optional path to a pretrained MLP-AE ``checkpoint.pth`` (or its
        containing directory). When set, Stage 1 loads + freezes this AE instead
        of pretraining one on the fly — e.g. the upstream shipped checkpoints.
        ``d_model`` / ``d_ff`` / ``enc_in`` must match the checkpoint.
    """

    # Trainer convention: feed the raw future target each training step.
    requires_train_target = True

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        d_ff: int = 128,
        mse_weight: float = 10.0,
        cosine_weight: float = 15.0,
        use_latent_norm: bool = True,
        kernel_size: int = 25,
        individual: bool = False,
        ae_train_epochs: int = 100,
        ae_lr: float = 5e-4,
        ae_loss: str = "MAE",
        autoencoder_path: str = "",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.mse_weight = float(mse_weight)
        self.cosine_weight = float(cosine_weight)
        self.ae_train_epochs = int(ae_train_epochs)
        self.ae_lr = float(ae_lr)
        self.ae_loss = str(ae_loss).upper()
        # When set, Stage 1 loads this frozen AE checkpoint instead of
        # pretraining one on the fly (e.g. the upstream shipped checkpoints).
        self.autoencoder_path = str(autoencoder_path or "")

        self.ae = _MLPAutoEncoder(enc_in, d_model, d_ff)
        # The forecaster operates on the d_model-dim latent, so its channel
        # count is d_model rather than enc_in.
        self.backbone = DLinearBackbone(
            c_in=d_model,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=kernel_size,
            individual=individual,
        )
        self.latent_norm = (
            nn.LayerNorm(d_model, elementwise_affine=False) if use_latent_norm else None
        )

        self._ae_ready = False
        # Future target stashed by the trainer (set_train_target); consumed per forward.
        self._target: torch.Tensor | None = None
        # Read by the trainer after each forward; replaces the observation loss.
        self.train_loss_override: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Stage 1 — AE pretraining hook (run_one calls this once pre-training)
    # ------------------------------------------------------------------ #
    def pretrain(self, train_loader, device) -> None:
        """Provision the frozen autoencoder for Stage 1.

        Loads ``autoencoder_path`` when set; otherwise pretrains the AE by
        reconstruction. Idempotent: a second call (or ``ae_train_epochs <= 0``
        with no checkpoint) only freezes.
        """
        if self._ae_ready:
            return
        if self.autoencoder_path:
            self._load_pretrained_ae(device)
            return
        if self.ae_train_epochs <= 0:
            self._freeze_ae()
            return

        self.ae.to(device).train()
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.ae_lr)
        criterion = nn.L1Loss() if self.ae_loss == "MAE" else nn.MSELoss()
        print(
            f"[LatentTSF] Stage 1: pretraining AE "
            f"(epochs={self.ae_train_epochs}, lr={self.ae_lr}, loss={self.ae_loss})"
        )
        for epoch in range(self.ae_train_epochs):
            last = 0.0
            for batch in train_loader:
                x = batch[0].float().to(device)  # (B, seq_len, enc_in)
                optimizer.zero_grad()
                loss = criterion(self.ae(x), x)
                loss.backward()
                optimizer.step()
                last = loss.item()
            if (epoch + 1) % max(1, self.ae_train_epochs // 5) == 0:
                print(
                    f"[LatentTSF]   AE epoch {epoch + 1}/{self.ae_train_epochs} "
                    f"| recon {last:.6f}"
                )
        self._freeze_ae()

    def _load_pretrained_ae(self, device) -> None:
        """Load a frozen MLP-AE from ``autoencoder_path`` (file or directory)."""
        path = self.autoencoder_path
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LatentTSF autoencoder_path not found: {path!r}. Point it at a "
                "pretrained AE checkpoint.pth (or its folder), or clear it to "
                "pretrain the AE on the fly."
            )
        state = torch.load(path, map_location=device)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        # Tolerate a DataParallel 'module.' prefix.
        state = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        try:
            self.ae.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load LatentTSF AE from {path}: {exc}. Ensure "
                "d_model / d_ff / enc_in match the checkpoint (see the upstream "
                "'Pretrained AE Checkpoints' table)."
            ) from exc
        print(f"[LatentTSF] Stage 1: loaded pretrained AE from {path}")
        self._freeze_ae()

    def _freeze_ae(self) -> None:
        for param in self.ae.parameters():
            param.requires_grad = False
        # Drop any grads left from pretraining so a frozen param can never be
        # stepped by the Stage-2 optimizer.
        self.ae.zero_grad(set_to_none=True)
        self.ae.eval()
        self._ae_ready = True

    def train(self, mode: bool = True):
        """Keep the frozen AE in eval mode even when the parent switches to train."""
        super().train(mode)
        if self._ae_ready:
            self.ae.eval()
        return self

    # ------------------------------------------------------------------ #
    # Trainer convention — receive the raw future target for the latent loss
    # ------------------------------------------------------------------ #
    def set_train_target(self, y: torch.Tensor | None) -> None:
        self._target = y

    # ------------------------------------------------------------------ #
    # Stage 2 — latent-space forecasting
    # ------------------------------------------------------------------ #
    def _encode(self, x: torch.Tensor, norm: bool) -> torch.Tensor:
        z = self.ae.encode(x)
        if norm and self.latent_norm is not None:
            z = self.latent_norm(z)
        return z

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        # x: (B, seq_len, enc_in)
        z_x = self._encode(x, norm=True)               # (B, seq_len, d_model)
        z_pred = self.backbone(z_x)                     # (B, pred_len, d_model)
        z_pred = z_pred[:, -self.pred_len:, :]
        y_hat = self.ae.decode(z_pred)                  # (B, pred_len, enc_in)

        # Training-only latent objective. Gated on a target being fed (the
        # trainer feeds it only during training, never during validation), so
        # this is independent of the module's train/eval flag.
        self.train_loss_override = None
        if self._target is not None:
            y_true = self._target[:, -self.pred_len:, :].float().to(z_pred.device)
            with torch.no_grad():  # frozen encoder -> Z_Y is a constant target
                z_y = self.ae.encode(y_true)            # raw latent (no LayerNorm)
            mse = F.mse_loss(z_pred, z_y)
            cos = _cosine_align_loss(z_pred, z_y)
            self.train_loss_override = self.mse_weight * mse + self.cosine_weight * cos
            self._target = None  # consume one-shot
        return y_hat
