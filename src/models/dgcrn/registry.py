"""Model registration for DGCRN."""

from benchmark.registry import MODEL_REGISTRY
from models.dgcrn.model import Model
from models.dgcrn.schema import ModelParameterConfig


def register() -> None:
    """Register the DGCRN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DGCRN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            gcn_depth=params.get("gcn_depth", 1),
            rnn_size=params.get("rnn_size", 16),
            node_dim=params.get("node_dim", 8),
            hyper_gnn_dim=params.get("hyper_gnn_dim", 8),
            middle_dim=params.get("middle_dim", 2),
            subgraph_size=params.get("subgraph_size", 20),
            tanhalpha=params.get("tanhalpha", 3.0),
            dropout=params.get("dropout", 0.3),
        ),
        ModelParameterConfig,
    )
