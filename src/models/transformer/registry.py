"""Model registration for the vanilla Transformer."""

from benchmark.registry import MODEL_REGISTRY
from models.transformer.model import Model
from models.transformer.schema import ModelParameterConfig


def register() -> None:
    """Register Transformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Transformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            dec_in=params.get("dec_in"),
            c_out=params.get("c_out"),
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_layers=params.get("d_layers", 1),
            d_ff=params.get("d_ff", 256),
            dropout=params.get("dropout", 0.1),
            factor=params.get("factor", 3),
            activation=params.get("activation", "gelu"),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
