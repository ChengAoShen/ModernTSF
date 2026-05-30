"""Model registration for Crossformer."""

from benchmark.registry import MODEL_REGISTRY
from models.crossformer.model import Model
from models.crossformer.schema import ModelParameterConfig


def register() -> None:
    """Register Crossformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Crossformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 4),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 128),
            seg_len=params.get("seg_len", 12),
            win_size=params.get("win_size", 2),
            factor=params.get("factor", 10),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
