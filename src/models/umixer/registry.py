"""Model registration for U-Mixer."""

from benchmark.registry import MODEL_REGISTRY
from models.umixer.model import Model
from models.umixer.schema import ModelParameterConfig


def register() -> None:
    """Register U-Mixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "UMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out") or params["enc_in"],
            d_model=params.get("d_model", 64),
            e_layers=params.get("e_layers", 2),
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
