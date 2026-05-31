"""Model registration for PM25_GNN."""

from benchmark.registry import MODEL_REGISTRY
from models.pm25gnn.model import Model
from models.pm25gnn.schema import ModelParameterConfig


def register() -> None:
    """Register the PM25_GNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PM25_GNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            hid_dim=params.get("hid_dim", 64),
        ),
        ModelParameterConfig,
    )
