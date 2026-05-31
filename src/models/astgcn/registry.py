"""Model registration for ASTGCN."""

from benchmark.registry import MODEL_REGISTRY
from models.astgcn.model import Model
from models.astgcn.schema import ModelParameterConfig


def register() -> None:
    """Register the ASTGCN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "ASTGCN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            nb_block=params.get("nb_block", 2),
            K=params.get("K", 3),
            nb_chev_filter=params.get("nb_chev_filter", 64),
            nb_time_filter=params.get("nb_time_filter", 64),
        ),
        ModelParameterConfig,
    )
