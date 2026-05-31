"""Model registration for MGSFformer."""

from benchmark.registry import MODEL_REGISTRY
from models.mgsfformer.model import Model
from models.mgsfformer.schema import ModelParameterConfig


def register() -> None:
    """Register the MGSFformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MGSFformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            IE_dim=params.get("IE_dim", 32),
            dropout=params.get("dropout", 0.3),
            num_head=params.get("num_head", 2),
        ),
        ModelParameterConfig,
    )
