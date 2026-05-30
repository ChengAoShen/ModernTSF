"""Model registration for MultiPatchFormer."""

from benchmark.registry import MODEL_REGISTRY
from models.multipatchformer.model import Model
from models.multipatchformer.schema import ModelParameterConfig


def register() -> None:
    """Register MultiPatchFormer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MultiPatchFormer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 4),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 128),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
