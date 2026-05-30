"""Model registration for Fredformer."""

from benchmark.registry import MODEL_REGISTRY
from models.fredformer.model import Model
from models.fredformer.schema import ModelParameterConfig


def register() -> None:
    """Register Fredformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Fredformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 16),
            e_layers=params.get("e_layers", 2),
            n_heads=params.get("n_heads", 8),
            d_ff=params.get("d_ff", 128),
            dropout=params.get("dropout", 0.1),
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", True)),
            subtract_last=bool(params.get("subtract_last", False)),
            individual=bool(params.get("individual", False)),
            head_dropout=params.get("head_dropout", 0.0),
            cf_dim=params.get("cf_dim", 48),
            cf_depth=params.get("cf_depth", 2),
            cf_heads=params.get("cf_heads", 6),
            cf_mlp=params.get("cf_mlp", 128),
            cf_head_dim=params.get("cf_head_dim", 32),
            cf_drop=params.get("cf_drop", 0.2),
        ),
        ModelParameterConfig,
    )
