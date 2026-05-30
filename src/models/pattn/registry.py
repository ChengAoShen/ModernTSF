"""Model registration for PAttn."""

from benchmark.registry import MODEL_REGISTRY
from models.pattn.model import Model
from models.pattn.schema import ModelParameterConfig


def register() -> None:
    """Register PAttn model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PAttn",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            d_ff=params.get("d_ff", 256),
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            dropout=params.get("dropout", 0.1),
            factor=params.get("factor", 3),
            activation=params.get("activation", "gelu"),
        ),
        ModelParameterConfig,
    )
