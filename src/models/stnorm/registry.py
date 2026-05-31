"""Model registration for STNorm."""

from benchmark.registry import MODEL_REGISTRY
from models.stnorm.model import Model
from models.stnorm.schema import ModelParameterConfig


def register() -> None:
    """Register the STNorm model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STNorm",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim"),
            channels=params.get("channels", 16),
            kernel_size=params.get("kernel_size", 2),
            blocks=params.get("blocks", 8),
            layers=params.get("layers", 2),
        ),
        ModelParameterConfig,
    )
