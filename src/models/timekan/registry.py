"""Model registration for TimeKAN."""

from benchmark.registry import MODEL_REGISTRY
from models.timekan.model import Model
from models.timekan.schema import ModelParameterConfig


def register() -> None:
    """Register TimeKAN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeKAN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out", None),
            d_model=params.get("d_model", 16),
            e_layers=params.get("e_layers", 1),
            down_sampling_window=params.get("down_sampling_window", 2),
            down_sampling_layers=params.get("down_sampling_layers", 1),
            begin_order=params.get("begin_order", 0),
            moving_avg=params.get("moving_avg", 25),
            dropout=params.get("dropout", 0.1),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
            use_norm=params.get("use_norm", 1),
        ),
        ModelParameterConfig,
    )
