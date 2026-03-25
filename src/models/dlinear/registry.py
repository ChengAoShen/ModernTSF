"""Model registration for DLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.dlinear.model import Model
from models.dlinear.schema import ModelParameterConfig


def register() -> None:
    """Register DLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DLinear",
        lambda cfg, params: Model(
            c_in=params["enc_in"],
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            kernel_size=params.get("kernel_size", 25),
            individual=params.get("individual", False),
        ),
        ModelParameterConfig,
    )
