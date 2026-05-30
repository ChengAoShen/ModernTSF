"""Model registration for BiST."""

from benchmark.registry import MODEL_REGISTRY
from models.bist.model import Model
from models.bist.schema import ModelParameterConfig


def register() -> None:
    """Register the BiST model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "BiST",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            model_dim=params.get("model_dim", 32),
            prompt_dim=params.get("prompt_dim", 32),
            num_layer=params.get("num_layer", 2),
            hid_dim=params.get("hid_dim", 64),
            tod_size=params.get("tod_size", 24),
            kernel_size=params.get("kernel_size", 3),
            rp_layer=params.get("rp_layer", 1),
            adaptive_adj_dim=params.get("adaptive_adj_dim", 10),
            core=params.get("core", 0),
        ),
        ModelParameterConfig,
    )
