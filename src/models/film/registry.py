"""Model registration for FiLM."""

from benchmark.registry import MODEL_REGISTRY
from models.film.model import Model
from models.film.schema import ModelParameterConfig


def register() -> None:
    """Register FiLM model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "FiLM",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            e_layers=params.get("e_layers", 2),
            ratio=params.get("ratio", 0.5),
            multiscale=params.get("multiscale", [1, 2, 4]),
            window_size=params.get("window_size", [256]),
        ),
        ModelParameterConfig,
    )
