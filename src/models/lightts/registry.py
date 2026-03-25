"""Model registration for LightTS."""

from benchmark.registry import MODEL_REGISTRY
from models.lightts.model import Model
from models.lightts.schema import ModelParameterConfig


def register() -> None:
    """Register LightTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "LightTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            hid_dim=params.get("hid_dim", 128),
            dropout=params.get("dropout", 0.0),
            chunk_size=params.get("chunk_size", 40),
            c_dim=params.get("c_dim", 40),
        ),
        ModelParameterConfig,
    )
