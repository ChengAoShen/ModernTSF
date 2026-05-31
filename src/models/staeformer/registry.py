"""Model registration for STAEformer."""

from benchmark.registry import MODEL_REGISTRY
from models.staeformer.model import Model
from models.staeformer.schema import ModelParameterConfig


def register() -> None:
    """Register the STAEformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STAEformer",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim"),
            input_embedding_dim=params.get("input_embedding_dim", 24),
            tod_embedding_dim=params.get("tod_embedding_dim", 24),
            dow_embedding_dim=params.get("dow_embedding_dim", 24),
            adaptive_embedding_dim=params.get(
                "adaptive_embedding_dim", 56),
            feed_forward_dim=params.get("feed_forward_dim", 128),
            num_heads=params.get("num_heads", 4),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            steps_per_day=params.get("steps_per_day", 24),
        ),
        ModelParameterConfig,
    )
