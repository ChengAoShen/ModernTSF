"""Model registration for Reformer."""

from benchmark.registry import MODEL_REGISTRY
from models.reformer.model import Model
from models.reformer.schema import ModelParameterConfig


def register() -> None:
    """Register Reformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Reformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out"),
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 256),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "gelu"),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
            bucket_size=params.get("bucket_size", 4),
            n_hashes=params.get("n_hashes", 4),
        ),
        ModelParameterConfig,
    )
