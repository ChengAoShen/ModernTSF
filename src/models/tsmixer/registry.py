"""Model registration for TSMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.tsmixer.model import Model
from models.tsmixer.schema import ModelParameterConfig


def register() -> None:
    """Register TSMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TSMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            e_layers=params.get("e_layers", 2),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
