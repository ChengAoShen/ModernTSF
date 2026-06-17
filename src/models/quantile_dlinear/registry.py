"""Model registration for QuantileDLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.quantile_dlinear.model import Model
from models.quantile_dlinear.schema import ModelParameterConfig


def register() -> None:
    """Register QuantileDLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "QuantileDLinear",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            features=cfg.task.features,
            kernel_size=params.get("kernel_size", 25),
            individual=bool(params.get("individual", False)),
            quantile_levels=params.get("quantile_levels"),
        ),
        ModelParameterConfig,
    )
