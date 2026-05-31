"""Model registration for AirFormer."""

from benchmark.registry import MODEL_REGISTRY
from models.airformer.model import Model
from models.airformer.schema import ModelParameterConfig


def register() -> None:
    """Register the AirFormer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "AirFormer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim", 2),
            d_model=params.get("d_model", 32),
            nhead=params.get("nhead", 2),
            num_encoder_layers=params.get("num_encoder_layers", 4),
            dropout=params.get("dropout", 0.3),
        ),
        ModelParameterConfig,
    )
