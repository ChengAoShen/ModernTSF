"""Model registration for DSFormer."""

from benchmark.registry import MODEL_REGISTRY
from models.dsformer.model import Model
from models.dsformer.schema import ModelParameterConfig


def register() -> None:
    """Register the DSFormer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DSFormer",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            num_layer=params.get("num_layer", 1),
            dropout=params.get("dropout", 0.2),
            muti_head=params.get("muti_head", 4),
            num_samp=params.get("num_samp", 3),
        ),
        ModelParameterConfig,
    )
