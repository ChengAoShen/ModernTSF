"""Model registration for GAGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.gagnn.model import Model
from models.gagnn.schema import ModelParameterConfig


def register() -> None:
    """Register the GAGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "GAGNN",
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
        ),
        ModelParameterConfig,
    )
