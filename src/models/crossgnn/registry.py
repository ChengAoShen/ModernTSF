"""Model registration for CrossGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.crossgnn.model import Model
from models.crossgnn.schema import ModelParameterConfig


def register() -> None:
    """Register the CrossGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CrossGNN",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            seg_len=params.get("seg_len", 6),
            d_model=params.get("d_model", 128),
            d_ff=params.get("d_ff", 256),
            n_heads=params.get("n_heads", 4),
            e_layers=params.get("e_layers", 3),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
