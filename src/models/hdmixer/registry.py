"""Model registration for HDMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.hdmixer.model import Model
from models.hdmixer.schema import ModelParameterConfig


def register() -> None:
    """Register HDMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "HDMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            d_ff=params.get("d_ff", 256),
            e_layers=params.get("e_layers", 3),
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            dropout=params.get("dropout", 0.1),
            head_dropout=params.get("head_dropout", 0.0),
            activation=params.get("activation", "gelu"),
            individual=bool(params.get("individual", False)),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", True)),
            subtract_last=bool(params.get("subtract_last", False)),
            deform_range=params.get("deform_range", 0.25),
            mix_time=bool(params.get("mix_time", True)),
            mix_variable=bool(params.get("mix_variable", True)),
            mix_channel=bool(params.get("mix_channel", True)),
        ),
        ModelParameterConfig,
    )
