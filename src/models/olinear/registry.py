"""Model registration for OLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.olinear.model import Model
from models.olinear.schema import ModelParameterConfig


def register() -> None:
    """Register OLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "OLinear",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            dropout=params.get("dropout", 0.1),
            period=params.get("period", 24),
            num_prompts=params.get("num_prompts", 4),
            use_revin=bool(params.get("use_revin", True)),
        ),
        ModelParameterConfig,
    )
