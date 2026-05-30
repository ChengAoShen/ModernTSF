"""Model registration for HimNet."""

from benchmark.registry import MODEL_REGISTRY
from models.himnet.model import Model
from models.himnet.schema import ModelParameterConfig


def register() -> None:
    """Register the HimNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "HimNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            output_dim=params.get("output_dim", 1),
            hidden_dim=params.get("hidden_dim", 32),
            num_layers=params.get("num_layers", 1),
            cheb_k=params.get("cheb_k", 2),
            node_embedding_dim=params.get("node_embedding_dim", 8),
            st_embedding_dim=params.get("st_embedding_dim", 8),
            tod_embedding_dim=params.get("tod_embedding_dim", 8),
            dow_embedding_dim=params.get("dow_embedding_dim", 8),
            steps_per_day=params.get("steps_per_day", 288),
            use_teacher_forcing=params.get("use_teacher_forcing", True),
        ),
        ModelParameterConfig,
    )
