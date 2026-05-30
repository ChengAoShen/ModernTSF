"""Model registration for WaveNet."""

from benchmark.registry import MODEL_REGISTRY
from models.wavenet.model import Model
from models.wavenet.schema import ModelParameterConfig


def register() -> None:
    """Register WaveNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "WaveNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            residual_channels=params.get("residual_channels", 16),
            dilation_channels=params.get("dilation_channels", 16),
            skip_channels=params.get("skip_channels", 64),
            end_channels=params.get("end_channels", 128),
            kernel_size=params.get("kernel_size", 2),
            blocks=params.get("blocks", 2),
            layers=params.get("layers", 2),
            use_norm=bool(params.get("use_norm", True)),
        ),
        ModelParameterConfig,
    )
