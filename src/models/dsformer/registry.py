"""Model registration for DSFormer."""

from benchmark.registry import MODEL_REGISTRY
from models.dsformer.model import Model
from models.dsformer.schema import ModelParameterConfig


def register() -> None:
    """Register DSFormer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DSFormer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            num_layer=params.get("num_layer", 1),
            muti_head=params.get("muti_head", 2),
            num_samp=params.get("num_samp", 2),
            dropout=params.get("dropout", 0.15),
            if_node=bool(params.get("if_node", True)),
        ),
        ModelParameterConfig,
    )
