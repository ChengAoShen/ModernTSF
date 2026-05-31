"""Model registration for UMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.umixer.model import Model
from models.umixer.schema import ModelParameterConfig


def register() -> None:
    """Register the UMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "UMixer",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            patch_len=params.get("patch_len", 24),
            stride=params.get("stride", 24),
            d_model=params.get("d_model", 128),
            dropout=params.get("dropout", 0.1),
            e_layers=params.get("e_layers", 2),
            d_layers=params.get("d_layers", 1),
        ),
        ModelParameterConfig,
    )
