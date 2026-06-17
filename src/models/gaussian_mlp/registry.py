"""Model registration for GaussianMLP."""

from benchmark.registry import MODEL_REGISTRY
from models.gaussian_mlp.model import Model
from models.gaussian_mlp.schema import ModelParameterConfig


def register() -> None:
    """Register GaussianMLP model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "GaussianMLP",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            features=cfg.task.features,
            hidden_size=params.get("hidden_size", 256),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            eps=params.get("eps", 1e-6),
        ),
        ModelParameterConfig,
    )
