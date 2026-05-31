"""Model registration for STGODE."""

from benchmark.registry import MODEL_REGISTRY
from models.stgode.model import Model
from models.stgode.schema import ModelParameterConfig


def register() -> None:
    """Register the STGODE model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STGODE",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            num_layers=params.get("num_layers", 3),
        ),
        ModelParameterConfig,
    )
