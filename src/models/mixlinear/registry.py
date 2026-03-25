"""Model registration for MixLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.mixlinear.model import Model
from models.mixlinear.schema import ModelParameterConfig


def register() -> None:
    """Register MixLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MixLinear",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            period_len=params.get("period_len", 24),
            com_len=params.get("com_len", 4),
            lpf=params.get("lpf", 1),
            alpha=params.get("alpha", 0.5),
        ),
        ModelParameterConfig,
    )
