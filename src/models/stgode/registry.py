"""Model registration for STGODE."""

from benchmark.registry import MODEL_REGISTRY
from models.stgode.model import Model
from models.stgode.schema import ModelParameterConfig


def register() -> None:
    """Register the STGODE model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STGODE",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
        ),
        ModelParameterConfig,
    )
