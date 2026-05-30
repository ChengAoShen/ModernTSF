"""Model registration for BigST."""

from benchmark.registry import MODEL_REGISTRY
from models.bigst.model import Model
from models.bigst.schema import ModelParameterConfig


def register() -> None:
    """Register the BigST model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "BigST",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            hid_dim=params.get("hid_dim", 16),
            node_dim=params.get("node_dim", 8),
            time_dim=params.get("time_dim", 8),
            tod_size=params.get("tod_size", 24),
            dow_size=params.get("dow_size", 7),
            tau=params.get("tau", 1.0),
            random_feature_dim=params.get("random_feature_dim", 16),
            dropout=params.get("dropout", 0.1),
            use_residual=params.get("use_residual", True),
            use_bn=params.get("use_bn", True),
        ),
        ModelParameterConfig,
    )
