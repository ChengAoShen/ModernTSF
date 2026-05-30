"""Model registration for DUET."""

from benchmark.registry import MODEL_REGISTRY
from models.duet.model import Model
from models.duet.schema import ModelParameterConfig


def register() -> None:
    """Register DUET model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DUET",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 4),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 64),
            dropout=params.get("dropout", 0.1),
            fc_dropout=params.get("fc_dropout", 0.1),
            factor=params.get("factor", 3),
            activation=params.get("activation", "gelu"),
            moving_avg=params.get("moving_avg", 25),
            num_experts=params.get("num_experts", 4),
            k=params.get("k", 2),
            hidden_size=params.get("hidden_size", 64),
            noisy_gating=bool(params.get("noisy_gating", True)),
            CI=bool(params.get("CI", True)),
            output_attention=bool(params.get("output_attention", False)),
        ),
        ModelParameterConfig,
    )
