"""Model registration for STID."""

from benchmark.registry import MODEL_REGISTRY
from models.stid.model import Model
from models.stid.schema import ModelParameterConfig


def register() -> None:
    """Register the STID model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STID",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            embed_dim=params.get("embed_dim", 32),
            num_layers=params.get("num_layers", 1),
            num_time_in_day=params.get("num_time_in_day", 24),
            num_day_in_week=params.get("num_day_in_week", 7),
            if_time_in_day=params.get("if_time_in_day", True),
            if_day_in_week=params.get("if_day_in_week", True),
        ),
        ModelParameterConfig,
    )
