"""Model registration for DeepAR."""

from benchmark.registry import MODEL_REGISTRY
from models.deepar.model import Model
from models.deepar.schema import ModelParameterConfig


def register() -> None:
    """Register DeepAR model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DeepAR",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            embedding_size=params.get("embedding_size", 32),
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
            cov_feat_size=params.get("cov_feat_size", 0),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
