"""Model registration for STGCN."""

from benchmark.registry import MODEL_REGISTRY
from models.stgcn.model import Model
from models.stgcn.schema import ModelParameterConfig


def register() -> None:
    """Register the STGCN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STGCN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            Kt=params.get("Kt", 3),
            Ks=params.get("Ks", 3),
            hidden_dim=params.get("hidden_dim", 64),
            bottleneck_dim=params.get("bottleneck_dim", 16),
            out_hidden_dim=params.get("out_hidden_dim", 128),
            act_func=params.get("act_func", "glu"),
            graph_conv_type=params.get("graph_conv_type", "cheb_graph_conv"),
            bias=params.get("bias", True),
            droprate=params.get("droprate", 0.5),
        ),
        ModelParameterConfig,
    )
