"""Model registration for PCDCNet."""

from benchmark.registry import MODEL_REGISTRY
from models.pcdcnet.model import Model
from models.pcdcnet.schema import ModelParameterConfig


def register() -> None:
    """Register the PCDCNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PCDCNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim"),
            d_model=params.get("d_model", 64),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
