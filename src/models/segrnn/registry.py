"""Model registration for SegRNN."""

from benchmark.registry import MODEL_REGISTRY
from models.segrnn.model import Model
from models.segrnn.schema import ModelParameterConfig


def register() -> None:
    """Register SegRNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "SegRNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            dropout=params.get("dropout", 0.1),
            seg_len=params.get("seg_len", 24),
        ),
        ModelParameterConfig,
    )
