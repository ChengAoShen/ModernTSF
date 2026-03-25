"""Model registration for Autoformer."""

from benchmark.registry import MODEL_REGISTRY
from models.autoformer.model import Model
from models.autoformer.schema import ModelParameterConfig


def register() -> None:
    """Register Autoformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Autoformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            label_len=cfg.task.label_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            dec_in=params["dec_in"],
            c_out=params["c_out"],
            d_model=params.get("d_model", 512),
            n_heads=params.get("n_heads", 8),
            e_layers=params.get("e_layers", 2),
            d_layers=params.get("d_layers", 1),
            d_ff=params.get("d_ff", 2048),
            moving_avg=params.get("moving_avg", 25),
            factor=params.get("factor", 1),
            freq=params.get("freq", "h"),
            dropout=params.get("dropout", 0.1),
            embed=params.get("embed", "timeF"),
            num_class=params.get("num_class", 7),
            activation=params.get("activation", "gelu"),
            task_name=params.get("task_name", "long_term_forecast"),
        ),
        ModelParameterConfig,
    )
