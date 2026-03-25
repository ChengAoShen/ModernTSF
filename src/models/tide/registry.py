"""Model registration for TiDE."""

from benchmark.registry import MODEL_REGISTRY
from models.tide.model import Model
from models.tide.schema import ModelParameterConfig


def register() -> None:
    """Register TiDE model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TiDE",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            d_model=params.get("d_model", 512),
            e_layers=params.get("e_layers", 2),
            d_layers=params.get("d_layers", 1),
            d_ff=params.get("d_ff", 2048),
            c_out=params.get("c_out", 7),
            freq=params.get("freq", "h"),
            dropout=params.get("dropout", 0.1),
            bias=bool(params.get("bias", True)),
            feature_encode_dim=params.get("feature_encode_dim", 2),
        ),
        ModelParameterConfig,
    )
