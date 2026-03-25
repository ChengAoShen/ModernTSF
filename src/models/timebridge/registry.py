"""Model registration for TimeBridge."""

from benchmark.registry import MODEL_REGISTRY
from models.timebridge.model import Model
from models.timebridge.schema import ModelParameterConfig


def register() -> None:
    """Register TimeBridge model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeBridge",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            period=params.get("period", 24),
            num_p=params.get("num_p"),
            ia_layers=params.get("ia_layers", 2),
            pd_layers=params.get("pd_layers", 1),
            ca_layers=params.get("ca_layers", 2),
            stable_len=params.get("stable_len", 3),
            d_model=params.get("d_model", 16),
            n_heads=params.get("n_heads", 4),
            d_ff=params.get("d_ff", 128),
            attn_dropout=params.get("attn_dropout", 0.15),
            dropout=params.get("dropout", 0.0),
            activation=params.get("activation", "gelu"),
            revin=bool(params.get("revin", True)),
            time_feat_dim=params.get("time_feat_dim", 6),
        ),
        ModelParameterConfig,
    )
