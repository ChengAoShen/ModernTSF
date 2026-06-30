"""Model registration for LatentTSF."""

from benchmark.registry import MODEL_REGISTRY
from models.latenttsf.model import Model
from models.latenttsf.schema import ModelParameterConfig


def register() -> None:
    """Register LatentTSF model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "LatentTSF",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            d_ff=params.get("d_ff", 128),
            mse_weight=params.get("mse_weight", 10.0),
            cosine_weight=params.get("cosine_weight", 15.0),
            use_latent_norm=bool(params.get("use_latent_norm", True)),
            kernel_size=params.get("kernel_size", 25),
            individual=bool(params.get("individual", False)),
            ae_train_epochs=params.get("ae_train_epochs", 100),
            ae_lr=params.get("ae_lr", 5e-4),
            ae_loss=params.get("ae_loss", "MAE"),
            autoencoder_path=params.get("autoencoder_path", ""),
        ),
        ModelParameterConfig,
    )
