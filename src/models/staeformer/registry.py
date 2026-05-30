"""Model registration for STAEformer."""

from benchmark.registry import MODEL_REGISTRY
from models.staeformer.model import Model
from models.staeformer.schema import ModelParameterConfig


def register() -> None:
    """Register the STAEformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STAEformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            steps_per_day=params.get("steps_per_day", 24),
            input_embedding_dim=params.get("input_embedding_dim", 24),
            tod_embedding_dim=params.get("tod_embedding_dim", 24),
            dow_embedding_dim=params.get("dow_embedding_dim", 24),
            spatial_embedding_dim=params.get("spatial_embedding_dim", 0),
            adaptive_embedding_dim=params.get("adaptive_embedding_dim", 80),
            feed_forward_dim=params.get("feed_forward_dim", 256),
            num_heads=params.get("num_heads", 4),
            num_layers=params.get("num_layers", 3),
            dropout=params.get("dropout", 0.1),
            use_mixed_proj=params.get("use_mixed_proj", True),
        ),
        ModelParameterConfig,
    )
