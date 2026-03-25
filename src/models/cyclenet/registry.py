"""Model registration for CycleNet."""

from benchmark.registry import MODEL_REGISTRY
from models.cyclenet.model import Model
from models.cyclenet.schema import ModelParameterConfig


def register() -> None:
    """Register CycleNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CycleNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cycle=params.get("cycle", 24),
            model_type=params.get("model_type", "linear"),
            d_model=params.get("d_model", 512),
            use_revin=bool(params.get("use_revin", True)),
        ),
        ModelParameterConfig,
    )
