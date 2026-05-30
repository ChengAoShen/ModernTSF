"""Model registration for ETSformer."""

from benchmark.registry import MODEL_REGISTRY
from models.etsformer.model import Model
from models.etsformer.schema import ModelParameterConfig


def register() -> None:
    """Register ETSformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "ETSformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out", None),
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_layers=params.get("d_layers", 2),
            d_ff=params.get("d_ff", 256),
            top_k=params.get("top_k", 3),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "sigmoid"),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
