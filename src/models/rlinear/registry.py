"""Model registration for RLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.rlinear.model import Model
from models.rlinear.schema import ModelParameterConfig


def register() -> None:
    """Register RLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "RLinear",
        lambda cfg, params: Model(
            c_in=params["enc_in"],
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            individual=bool(params.get("individual", False)),
            affine=bool(params.get("affine", False)),
            subtract_last=bool(params.get("subtract_last", False)),
        ),
        ModelParameterConfig,
    )
