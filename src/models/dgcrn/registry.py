"""Model registration for DGCRN."""

from benchmark.registry import MODEL_REGISTRY
from models.dgcrn.model import Model
from models.dgcrn.schema import ModelParameterConfig


def register() -> None:
    """Register the DGCRN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DGCRN",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            gcn_depth=params.get("gcn_depth", 2),
            dropout=params.get("dropout", 0.3),
            subgraph_size=params.get("subgraph_size", 20),
            node_dim=params.get("node_dim", 40),
            rnn_size=params.get("rnn_size", 64),
            adj_type=params.get("adj_type", "doubletransition"),
        ),
        ModelParameterConfig,
    )
