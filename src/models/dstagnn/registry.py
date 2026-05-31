"""Model registration for DSTAGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.dstagnn.model import Model
from models.dstagnn.schema import ModelParameterConfig


def register() -> None:
    """Register the DSTAGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DSTAGNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            d_model=params.get("d_model", 64),
            d_k=params.get("d_k", 8),
            d_v=params.get("d_v", 8),
            n_heads=params.get("n_heads", 4),
        ),
        ModelParameterConfig,
    )
