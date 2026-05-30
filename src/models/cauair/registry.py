"""Model registration for CauAir."""

from benchmark.registry import MODEL_REGISTRY
from models.cauair.model import Model
from models.cauair.schema import ModelParameterConfig


def register() -> None:
    """Register the CauAir model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CauAir",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim"),
            dim=params.get("dim", 64),
            rank=params.get("rank", 8),
            head=params.get("head", 4),
        ),
        ModelParameterConfig,
    )
