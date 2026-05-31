"""Model registration for HL."""

from benchmark.registry import MODEL_REGISTRY
from models.hl.model import Model
from models.hl.schema import ModelParameterConfig


def register() -> None:
    """Register the HL model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "HL",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
        ),
        ModelParameterConfig,
    )
