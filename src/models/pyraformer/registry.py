"""Model registration for Pyraformer."""

from benchmark.registry import MODEL_REGISTRY
from models.pyraformer.model import Model
from models.pyraformer.schema import ModelParameterConfig


def register() -> None:
    """Register Pyraformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Pyraformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 256),
            dropout=params.get("dropout", 0.1),
            window_size=params.get("window_size", [4, 4]),
            inner_size=params.get("inner_size", 5),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
