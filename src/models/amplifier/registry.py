"""Model registration for Amplifier."""

from benchmark.registry import MODEL_REGISTRY
from models.amplifier.model import Model
from models.amplifier.schema import ModelParameterConfig


def register() -> None:
    """Register Amplifier model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Amplifier",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            hidden_size=params.get("hidden_size", 128),
            sci=bool(params.get("sci", False)),
        ),
        ModelParameterConfig,
    )
