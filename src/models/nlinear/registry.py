"""Model registration for NLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.nlinear.model import Model
from models.nlinear.schema import ModelParameterConfig


def register() -> None:
    """Register NLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "NLinear",
        lambda cfg, params: Model(
            c_in=params["enc_in"],
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            individual=bool(params.get("individual", False)),
        ),
        ModelParameterConfig,
    )
