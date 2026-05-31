"""Model registration for DeepAir."""

from benchmark.registry import MODEL_REGISTRY
from models.deepair.model import Model
from models.deepair.schema import ModelParameterConfig


def register() -> None:
    """Register the DeepAir model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DeepAir",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim", 2),
            hid_dim=params.get("hid_dim", 64),
        ),
        ModelParameterConfig,
    )
