"""Model registration for AirCade."""

from benchmark.registry import MODEL_REGISTRY
from models.aircade.model import Model
from models.aircade.schema import ModelParameterConfig


def register() -> None:
    """Register the AirCade model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "AirCade",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim"),
            input_embedding_dim=params.get("input_embedding_dim", 16),
            adaptive_embedding_dim=params.get("adaptive_embedding_dim", 24),
            feed_forward_dim=params.get("feed_forward_dim", 64),
            num_heads=params.get("num_heads", 4),
            num_layers=params.get("num_layers", 1),
            node_embed_dim=params.get("node_embed_dim", 10),
        ),
        ModelParameterConfig,
    )
