"""Model registration for xPatch."""

from benchmark.registry import MODEL_REGISTRY
from models.xpatch.model import Model
from models.xpatch.schema import ModelParameterConfig


def register() -> None:
    """Register xPatch model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "xPatch",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            padding_patch=params.get("padding_patch", "end"),
            ma_type=params.get("ma_type", "ema"),
            alpha=params.get("alpha", 0.3),
            beta=params.get("beta", 0.3),
            revin=bool(params.get("revin", True)),
        ),
        ModelParameterConfig,
    )
