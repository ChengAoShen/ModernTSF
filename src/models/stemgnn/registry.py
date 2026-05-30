"""Model registration for StemGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.stemgnn.model import Model
from models.stemgnn.schema import ModelParameterConfig


def register() -> None:
    """Register the StemGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "StemGNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            multi_layer=params.get("multi_layer", 3),
            dropout_rate=params.get("dropout_rate", 0.5),
            leaky_rate=params.get("leaky_rate", 0.2),
        ),
        ModelParameterConfig,
    )
