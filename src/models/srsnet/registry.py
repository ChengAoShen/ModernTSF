"""Model registration for SRSNet."""

from benchmark.registry import MODEL_REGISTRY
from models.srsnet.model import Model
from models.srsnet.schema import ModelParameterConfig


def register() -> None:
    """Register SRSNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "SRSNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            patch_len=params.get("patch_len", 24),
            stride=params.get("stride", 24),
            hidden_size=params.get("hidden_size", 64),
            dropout=params.get("dropout", 0.2),
            head_dropout=params.get("head_dropout", 0.1),
            alpha=params.get("alpha", 2.0),
            pos=bool(params.get("pos", True)),
            head_mode=params.get("head_mode", "linear"),
            affine=bool(params.get("affine", True)),
            subtract_last=bool(params.get("subtract_last", False)),
        ),
        ModelParameterConfig,
    )
