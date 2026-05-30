"""Model registration for WPMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.wpmixer.model import Model
from models.wpmixer.schema import ModelParameterConfig


def register() -> None:
    """Register WPMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "WPMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out"),
            d_model=params.get("d_model", 128),
            dropout=params.get("dropout", 0.1),
            tfactor=params.get("tfactor", 5),
            dfactor=params.get("dfactor", 5),
            wavelet=params.get("wavelet", "db2"),
            level=params.get("level", 1),
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            no_decomposition=bool(params.get("no_decomposition", False)),
        ),
        ModelParameterConfig,
    )
