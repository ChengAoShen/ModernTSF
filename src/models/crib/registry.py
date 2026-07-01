"""Model registration for CRIB."""

from benchmark.registry import MODEL_REGISTRY
from models.crib.model import Model
from models.crib.schema import ModelParameterConfig


def register() -> None:
    """Register CRIB model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CRIB",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            patch_len=params.get("patch_len", 8),
            model_dim=params.get("model_dim", 32),
            heads_num=params.get("heads_num", 4),
            enc_num=params.get("enc_num", 3),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "relu"),
            consis_weight=params.get("consis_weight", 1.0),
            kl_weight=params.get("kl_weight", 1e-6),
        ),
        ModelParameterConfig,
    )
