"""Model registration for RPMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.rpmixer.model import Model
from models.rpmixer.schema import ModelParameterConfig


def register() -> None:
    """Register the RPMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "RPMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim"),
            IE_dim=params.get("IE_dim", 32),
            dropout=params.get("dropout", 0.3),
            num_head=params.get("num_head", 2),
        ),
        ModelParameterConfig,
    )
