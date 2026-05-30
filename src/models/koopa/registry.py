"""Model registration for Koopa."""

from benchmark.registry import MODEL_REGISTRY
from models.koopa.model import Model
from models.koopa.schema import ModelParameterConfig


def register() -> None:
    """Register Koopa model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Koopa",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            seg_len=params.get("seg_len", None),
            dynamic_dim=params.get("dynamic_dim", 128),
            hidden_dim=params.get("hidden_dim", 64),
            hidden_layers=params.get("hidden_layers", 2),
            num_blocks=params.get("num_blocks", 3),
            multistep=bool(params.get("multistep", False)),
            alpha=params.get("alpha", 0.2),
        ),
        ModelParameterConfig,
    )
