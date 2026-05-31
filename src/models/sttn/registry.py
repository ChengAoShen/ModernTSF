"""Model registration for STTN."""

from benchmark.registry import MODEL_REGISTRY
from models.sttn.model import Model
from models.sttn.schema import ModelParameterConfig


def register() -> None:
    """Register the STTN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STTN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 4),
            num_layers=params.get("num_layers", 3),
            dropout=params.get("dropout", 0.1),
            adj_type=params.get("adj_type", "doubletransition"),
        ),
        ModelParameterConfig,
    )
