"""Model registration for STOP."""

from benchmark.registry import MODEL_REGISTRY
from models.stop.model import Model
from models.stop.schema import ModelParameterConfig


def register() -> None:
    """Register the STOP model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STOP",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            model_dim=params.get("model_dim", 16),
            prompt_dim=params.get("prompt_dim", 16),
            num_layer=params.get("num_layer", 2),
            hid_dim=params.get("hid_dim", 64),
            tod_size=params.get("tod_size", 24),
            kernel_size=params.get("kernel_size", 3),
            core=params.get("core", 4),
            head=params.get("head", 4),
        ),
        ModelParameterConfig,
    )
