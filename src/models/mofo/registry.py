"""Model registration for MoFo."""

from benchmark.registry import MODEL_REGISTRY
from models.mofo.model import Model
from models.mofo.schema import ModelParameterConfig


def register() -> None:
    """Register the MoFo model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MoFo",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            periodic=params.get("periodic", 24),
            head=params.get("head", 4),
            d_layers=params.get("d_layers", 1),
            bias=params.get("bias", 1),
            cias=params.get("cias", 1),
        ),
        ModelParameterConfig,
    )
