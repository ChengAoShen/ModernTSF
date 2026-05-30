"""Model registration for S_Mamba."""

from benchmark.registry import MODEL_REGISTRY
from models.s_mamba.model import Model
from models.s_mamba.schema import ModelParameterConfig


def register() -> None:
    """Register S_Mamba model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "S_Mamba",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 128),
            d_state=params.get("d_state", 16),
            d_ff=params.get("d_ff", 128),
            e_layers=params.get("e_layers", 2),
            d_conv=params.get("d_conv", 2),
            expand=params.get("expand", 1),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "gelu"),
            use_norm=bool(params.get("use_norm", True)),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
