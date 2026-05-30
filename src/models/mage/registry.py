"""Model registration for MAGE."""

from benchmark.registry import MODEL_REGISTRY
from models.mage.model import Model
from models.mage.schema import ModelParameterConfig


def register() -> None:
    """Register the MAGE model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MAGE",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            model_dim=params.get("model_dim", 64),
            recur_num=params.get("recur_num", 8),
            blocknum=params.get("blocknum", 3),
            topk=params.get("topk", 2),
            node_dim=params.get("node_dim", 16),
            tod_size=params.get("tod_size", 24),
        ),
        ModelParameterConfig,
    )
