"""Model registration for FreTS."""

from benchmark.registry import MODEL_REGISTRY
from models.frets.model import Model
from models.frets.schema import ModelParameterConfig


def register() -> None:
    """Register FreTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "FreTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            embed_size=params.get("embed_size", 128),
            hidden_size=params.get("hidden_size", 256),
            channel_independence=bool(params.get("channel_independence", False)),
        ),
        ModelParameterConfig,
    )
