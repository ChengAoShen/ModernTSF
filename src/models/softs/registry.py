"""Model registration for SOFTS."""

from benchmark.registry import MODEL_REGISTRY
from models.softs.model import Model
from models.softs.schema import ModelParameterConfig


def register() -> None:
    """Register SOFTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "SOFTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            d_core=params.get("d_core", 64),
            d_ff=params.get("d_ff", 256),
            e_layers=params.get("e_layers", 2),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "gelu"),
            use_norm=bool(params.get("use_norm", True)),
        ),
        ModelParameterConfig,
    )
