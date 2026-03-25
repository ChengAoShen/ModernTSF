"""Model registration for Linear."""

from benchmark.registry import MODEL_REGISTRY
from models.linear.model import Model
from models.linear.schema import ModelParameterConfig


def register() -> None:
    """Register Linear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Linear",
        lambda cfg, params: Model(
            c_in=params["enc_in"],
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            individual=bool(params.get("individual", False)),
        ),
        ModelParameterConfig,
    )
