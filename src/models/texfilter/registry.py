"""Model registration for TexFilter."""

from benchmark.registry import MODEL_REGISTRY
from models.texfilter.model import Model
from models.texfilter.schema import ModelParameterConfig


def register() -> None:
    """Register TexFilter model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TexFilter",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            embed_size=params.get("embed_size", 128),
            hidden_size=params.get("hidden_size", 256),
            dropout=params.get("dropout", 0.0),
        ),
        ModelParameterConfig,
    )
