"""Model registration for PaiFilter."""

from benchmark.registry import MODEL_REGISTRY
from models.paifilter.model import Model
from models.paifilter.schema import ModelParameterConfig


def register() -> None:
    """Register PaiFilter model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PaiFilter",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            hidden_size=params.get("hidden_size", 256),
        ),
        ModelParameterConfig,
    )
