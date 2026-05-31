"""Model registration for LSTM."""

from benchmark.registry import MODEL_REGISTRY
from models.lstm.model import Model
from models.lstm.schema import ModelParameterConfig


def register() -> None:
    """Register the LSTM model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "LSTM",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            init_dim=params.get("init_dim", 32),
            hid_dim=params.get("hid_dim", 64),
            end_dim=params.get("end_dim", 128),
            layer=params.get("layer", 2),
            dropout=params.get("dropout", 0.1),
            cov_dim=params.get("cov_dim", 2),
        ),
        ModelParameterConfig,
    )
