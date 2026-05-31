"""Model registration for TimeXer."""

from benchmark.registry import MODEL_REGISTRY
from models.timexer.model import Model
from models.timexer.schema import ModelParameterConfig


def register() -> None:
    """Register the TimeXer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeXer",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            patch_len=params.get("patch_len", 24),
            d_model=params.get("d_model", 128),
            dropout=params.get("dropout", 0.1),
            n_heads=params.get("n_heads", 4),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 256),
        ),
        ModelParameterConfig,
    )
