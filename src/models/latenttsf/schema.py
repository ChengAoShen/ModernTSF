from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Parameters for the faithful two-stage LatentTSF model."""

    enc_in: int
    # Latent state dimension and AE hidden width.
    d_model: int = 64
    d_ff: int = 128
    # Latent objective weights (paper Sec. 5.3.2): alpha (Pred), beta (Align).
    mse_weight: float = 10.0
    cosine_weight: float = 15.0
    use_latent_norm: bool = True
    # DLinear backbone (paper's primary Table-1 backbone) hyper-parameters.
    kernel_size: int = 25
    individual: bool = False
    # Stage-1 AE pretraining schedule (used when autoencoder_path is empty).
    # ae_loss is "MAE" (L1) or "MSE".
    ae_train_epochs: int = 100
    ae_lr: float = 5e-4
    ae_loss: str = "MAE"
    # Optional pretrained MLP-AE checkpoint.pth (or its folder). When set,
    # Stage 1 loads + freezes it instead of pretraining on the fly. d_model /
    # d_ff / enc_in must match the checkpoint.
    autoencoder_path: str = ""
