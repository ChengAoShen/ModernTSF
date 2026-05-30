"""Model registration for TimeXer."""

from benchmark.registry import MODEL_REGISTRY
from models.timexer.model import Model
from models.timexer.schema import ModelParameterConfig


def register() -> None:
    """Register TimeXer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeXer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 256),
            patch_len=params.get("patch_len", 16),
            dropout=params.get("dropout", 0.1),
            factor=params.get("factor", 3),
            activation=params.get("activation", "gelu"),
            use_norm=bool(params.get("use_norm", True)),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
