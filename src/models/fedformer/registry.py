"""Model registration for FEDformer."""

from benchmark.registry import MODEL_REGISTRY
from models.fedformer.model import Model
from models.fedformer.schema import ModelParameterConfig


def register() -> None:
    """Register FEDformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "FEDformer",
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
            freq=params.get("freq", "h"),
            dropout=params.get("dropout", 0.1),
            embed=params.get("embed", "timeF"),
            num_class=params.get("num_class", 7),
            activation=params.get("activation", "gelu"),
            task_name=params.get("task_name", "long_term_forecast"),
            version=params.get("version", "fourier"),
            mode_select=params.get("mode_select", "random"),
            modes=params.get("modes", 32),
        ),
        ModelParameterConfig,
    )
