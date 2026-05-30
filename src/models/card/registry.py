"""Model registration for CARD."""

from benchmark.registry import MODEL_REGISTRY
from models.card.model import Model
from models.card.schema import ModelParameterConfig


def register() -> None:
    """Register CARD model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CARD",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            d_model=params.get("d_model", 128),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 256),
            dropout=params.get("dropout", 0.1),
            dp_rank=params.get("dp_rank", 8),
            merge_size=params.get("merge_size", 2),
            momentum=params.get("momentum", 0.1),
            alpha=params.get("alpha", 0.5),
            use_statistic=bool(params.get("use_statistic", False)),
        ),
        ModelParameterConfig,
    )
