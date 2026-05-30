"""Model registration for MambaSimple."""

from benchmark.registry import MODEL_REGISTRY
from models.mambasimple.model import Model
from models.mambasimple.schema import ModelParameterConfig


def register() -> None:
    """Register MambaSimple model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MambaSimple",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out"),
            d_model=params.get("d_model", 128),
            d_ff=params.get("d_ff", 16),
            e_layers=params.get("e_layers", 2),
            expand=params.get("expand", 2),
            d_conv=params.get("d_conv", 4),
            dropout=params.get("dropout", 0.1),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
