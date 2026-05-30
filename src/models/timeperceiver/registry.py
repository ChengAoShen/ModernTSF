"""Model registration for TimePerceiver."""

from benchmark.registry import MODEL_REGISTRY
from models.timeperceiver.model import Model
from models.timeperceiver.schema import ModelParameterConfig


def register() -> None:
    """Register TimePerceiver model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimePerceiver",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 32),
            n_heads=params.get("n_heads", 2),
            d_ff=params.get("d_ff", 256),
            patch_len=params.get("patch_len", 16),
            dropout=params.get("dropout", 0.1),
            num_latents=params.get("num_latents", 8),
            latent_dim=params.get("latent_dim", 128),
            latent_d_ff=params.get("latent_d_ff", 256),
            num_latent_blocks=params.get("num_latent_blocks", 1),
            use_latent=bool(params.get("use_latent", True)),
            query_share=bool(params.get("query_share", True)),
        ),
        ModelParameterConfig,
    )
