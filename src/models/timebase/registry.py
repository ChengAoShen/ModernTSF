"""Model registration for TimeBase."""

from benchmark.registry import MODEL_REGISTRY
from models.timebase.model import Model
from models.timebase.schema import ModelParameterConfig


def register() -> None:
    """Register TimeBase model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeBase",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            period_len=params.get("period_len", 24),
            basis_num=params.get("basis_num", 6),
            individual=bool(params.get("individual", False)),
            use_orthogonal=bool(params.get("use_orthogonal", True)),
            use_period_norm=bool(params.get("use_period_norm", True)),
        ),
        ModelParameterConfig,
    )
