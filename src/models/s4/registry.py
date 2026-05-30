"""Model registration for S4 (S4D diagonal variant)."""

from benchmark.registry import MODEL_REGISTRY
from models.s4.model import Model
from models.s4.schema import ModelParameterConfig


def register() -> None:
    """Register S4 model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "S4",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            d_state=params.get("d_state", 64),
            e_layers=params.get("e_layers", 2),
            dropout=params.get("dropout", 0.1),
            use_norm=bool(params.get("use_norm", True)),
        ),
        ModelParameterConfig,
    )
