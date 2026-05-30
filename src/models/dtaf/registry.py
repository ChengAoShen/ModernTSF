"""Model registration for DTAF."""

from benchmark.registry import MODEL_REGISTRY
from models.dtaf.model import Model
from models.dtaf.schema import ModelParameterConfig


def register() -> None:
    """Register DTAF model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DTAF",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 32),
            e_layers=params.get("e_layers", 1),
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            heads=params.get("heads", 2),
            dropout=params.get("dropout", 0.1),
            moving_avg=params.get("moving_avg", 25),
            expert_num=params.get("expert_num", 2),
            kan_div=params.get("kan_div", 4),
            k=params.get("k", 1),
            aggregated_norm=params.get("aggregated_norm", 1),
        ),
        ModelParameterConfig,
    )
