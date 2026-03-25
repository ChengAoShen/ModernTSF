"""Model registration for SCINet."""

from benchmark.registry import MODEL_REGISTRY
from models.scinet.model import Model
from models.scinet.schema import ModelParameterConfig


def register() -> None:
    """Register SCINet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "SCINet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_layers=params.get("d_layers", 1),
            dropout=params.get("dropout", 0.0),
        ),
        ModelParameterConfig,
    )
