"""Model registration for TimeFilter."""

from benchmark.registry import MODEL_REGISTRY
from models.timefilter.model import Model
from models.timefilter.schema import ModelParameterConfig


def register() -> None:
    """Register TimeFilter model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeFilter",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            d_ff=params.get("d_ff", 128),
            n_heads=params.get("n_heads", 4),
            e_layers=params.get("e_layers", 2),
            patch_len=params.get("patch_len", 16),
            dropout=params.get("dropout", 0.1),
            alpha=params.get("alpha", 0.1),
            top_p=params.get("top_p", 0.5),
            pos=bool(params.get("pos", True)),
        ),
        ModelParameterConfig,
    )
