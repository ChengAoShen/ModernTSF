"""Model registration for CrossLinear."""

from benchmark.registry import MODEL_REGISTRY
from models.crosslinear.model import Model
from models.crosslinear.schema import ModelParameterConfig


def register() -> None:
    """Register CrossLinear model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CrossLinear",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            dec_in=params.get("dec_in") or params["enc_in"],
            patch_len=params.get("patch_len", 16),
            d_model=params.get("d_model", 32),
            d_ff=params.get("d_ff", 2048),
            alpha=params.get("alpha", 1.0),
            beta=params.get("beta", 0.5),
        ),
        ModelParameterConfig,
    )
