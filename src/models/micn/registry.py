"""Model registration for MICN."""

from benchmark.registry import MODEL_REGISTRY
from models.micn.model import Model
from models.micn.schema import ModelParameterConfig


def register() -> None:
    """Register MICN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MICN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out") or params["enc_in"],
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 4),
            d_layers=params.get("d_layers", 1),
            dropout=params.get("dropout", 0.05),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
            conv_kernel=params.get("conv_kernel", [12, 16]),
        ),
        ModelParameterConfig,
    )
