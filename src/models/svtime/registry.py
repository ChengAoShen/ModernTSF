"""Model registration for SVTime."""

from benchmark.registry import MODEL_REGISTRY
from models.svtime.model import Model
from models.svtime.schema import ModelParameterConfig


def register() -> None:
    """Register SVTime model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "SVTime",
        lambda cfg, params: Model(
            c_in=params["enc_in"],
            period=params.get("period", 24),
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            patch_size=params.get("patch_size", 6),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", False)),
            subtract_last=bool(params.get("subtract_last", False)),
            analysis_act=params.get("analysis_act", "relu"),
            analysis_hidden=params.get("analysis_hidden", "512,256"),
        ),
        ModelParameterConfig,
    )
