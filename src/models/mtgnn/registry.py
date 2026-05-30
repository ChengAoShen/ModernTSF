"""Model registration for MTGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.mtgnn.model import Model
from models.mtgnn.schema import ModelParameterConfig


def register() -> None:
    """Register the MTGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MTGNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            gcn_depth=params.get("gcn_depth", 2),
            subgraph_size=params.get("subgraph_size", 20),
            node_dim=params.get("node_dim", 40),
            conv_channels=params.get("conv_channels", 16),
            residual_channels=params.get("residual_channels", 16),
            skip_channels=params.get("skip_channels", 32),
            end_channels=params.get("end_channels", 64),
            layers=params.get("layers", 3),
            dropout=params.get("dropout", 0.3),
            propalpha=params.get("propalpha", 0.05),
            tanhalpha=params.get("tanhalpha", 3.0),
            dilation_exponential=params.get("dilation_exponential", 1),
            build_adj=params.get("build_adj", True),
        ),
        ModelParameterConfig,
    )
