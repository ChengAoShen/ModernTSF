"""Model registration for D2STGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.d2stgnn.model import Model
from models.d2stgnn.schema import ModelParameterConfig


def register() -> None:
    """Register the D2STGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "D2STGNN",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            d_model=params.get("d_model", 64),
            num_layers=params.get("num_layers", 4),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
