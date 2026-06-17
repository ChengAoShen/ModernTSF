"""Model registration for MQRNN."""

from benchmark.registry import MODEL_REGISTRY
from models.mqrnn.model import Model
from models.mqrnn.schema import ModelParameterConfig


def register() -> None:
    """Register MQRNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MQRNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            features=cfg.task.features,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 1),
            decoder_hidden=params.get("decoder_hidden", 64),
            dropout=params.get("dropout", 0.1),
            quantile_levels=params.get("quantile_levels"),
        ),
        ModelParameterConfig,
    )
