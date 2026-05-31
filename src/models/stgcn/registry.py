"""Model registration for STGCN."""

from benchmark.registry import MODEL_REGISTRY
from models.stgcn.model import Model
from models.stgcn.schema import ModelParameterConfig


def register() -> None:
    """Register the STGCN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STGCN",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            Ks=params.get("Ks", 3),
            Kt=params.get("Kt", 3),
            blocks=params.get("blocks"),
            drop_prob=params.get("drop_prob", 0.0),
        ),
        ModelParameterConfig,
    )
