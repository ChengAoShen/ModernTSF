"""Model registration for STDN."""

from benchmark.registry import MODEL_REGISTRY
from models.stdn.model import Model
from models.stdn.schema import ModelParameterConfig


def register() -> None:
    """Register the STDN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STDN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            time_slice_size=params.get("time_slice_size", 60),
            K=params.get("K", 4),
            d=params.get("d", 8),
            L=params.get("L", 1),
            order=params.get("order", 2),
            reference=params.get("reference", 4),
            out_channels=params.get("out_channels", 1),
        ),
        ModelParameterConfig,
    )
