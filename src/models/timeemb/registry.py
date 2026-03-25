"""Model registration for TimeEmb."""

from benchmark.registry import MODEL_REGISTRY
from models.timeemb.model import Model
from models.timeemb.schema import ModelParameterConfig


def register() -> None:
    """Register TimeEmb model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeEmb",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 512),
            use_revin=bool(params.get("use_revin", True)),
            use_hour_index=bool(params.get("use_hour_index", True)),
            use_day_index=bool(params.get("use_day_index", False)),
            scale=params.get("scale", 0.02),
            hour_length=params.get("hour_length", 24),
            day_length=params.get("day_length", 7),
        ),
        ModelParameterConfig,
    )
