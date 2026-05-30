"""Model registration for STNorm."""

from benchmark.registry import MODEL_REGISTRY
from models.stnorm.model import Model
from models.stnorm.schema import ModelParameterConfig


def register() -> None:
    """Register the STNorm model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STNorm",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            channels=params.get("channels", 16),
            kernel_size=params.get("kernel_size", 2),
            blocks=params.get("blocks", 2),
            layers=params.get("layers", 2),
            tnorm_bool=params.get("tnorm_bool", True),
            snorm_bool=params.get("snorm_bool", True),
        ),
        ModelParameterConfig,
    )
