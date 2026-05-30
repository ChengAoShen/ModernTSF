"""Model registration for BiMamba."""

from benchmark.registry import MODEL_REGISTRY
from models.bimamba.model import Model
from models.bimamba.schema import ModelParameterConfig


def register() -> None:
    """Register BiMamba model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "BiMamba",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out", None),
            d_model=params.get("d_model", 128),
            d_state=params.get("d_state", 16),
            e_layers=params.get("e_layers", 2),
            expand=params.get("expand", 2),
            d_conv=params.get("d_conv", 4),
            dropout=params.get("dropout", 0.1),
            share_ffn=bool(params.get("share_ffn", False)),
            share_norm=bool(params.get("share_norm", False)),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
