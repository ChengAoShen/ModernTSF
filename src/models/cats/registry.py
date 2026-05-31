"""Model registration for CATS."""

from benchmark.registry import MODEL_REGISTRY
from models.cats.model import Model
from models.cats.schema import ModelParameterConfig


def register() -> None:
    """Register the CATS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CATS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 16),
            d_ff=params.get("d_ff", 256),
            n_layers=params.get("n_layers", 3),
            dropout=params.get("dropout", 0.1),
            stride=params.get("stride", 24),
            QAM_start=params.get("QAM_start", 0.1),
            QAM_end=params.get("QAM_end", 0.5),
        ),
        ModelParameterConfig,
    )
