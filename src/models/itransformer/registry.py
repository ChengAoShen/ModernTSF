"""Model registration for iTransformer."""

from benchmark.registry import MODEL_REGISTRY
from models.itransformer.model import Model
from models.itransformer.schema import ModelParameterConfig


def register() -> None:
    """Register iTransformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "iTransformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            d_model=params.get("d_model", 512),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 2048),
            factor=params.get("factor", 1),
            dropout=params.get("dropout", 0.1),
            embed=params.get("embed", "timeF"),
            activation=params.get("activation", "gelu"),
            output_attention=bool(params.get("output_attention", False)),
            use_norm=bool(params.get("use_norm", True)),
            freq=params.get("freq", "h"),
            class_strategy=params.get("class_strategy", "projection"),
        ),
        ModelParameterConfig,
    )
