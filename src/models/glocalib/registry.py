"""Model registration for Glocal-IB."""

from benchmark.registry import MODEL_REGISTRY
from models.glocalib.model import Model
from models.glocalib.schema import ModelParameterConfig


def register() -> None:
    """Register Glocal-IB model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "GlocalIB",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            align_weight=params.get("align_weight", 0.5),
            mask_ratio=params.get("mask_ratio", 0.25),
            align_loss_type=params.get("align_loss_type", "cos_align"),
        ),
        ModelParameterConfig,
    )
