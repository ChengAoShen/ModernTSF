"""Model registration for TimeMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.timemixer.model import Model
from models.timemixer.schema import ModelParameterConfig


def register() -> None:
    """Register TimeMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            c_out=params["c_out"],
            e_layers=params.get("e_layers", 2),
            d_model=params.get("d_model", 512),
            d_ff=params.get("d_ff", 2048),
            down_sampling_window=params.get("down_sampling_window", 1),
            down_sampling_layers=params.get("down_sampling_layers", 0),
            down_sampling_method=params.get("down_sampling_method"),
            channel_independence=bool(params.get("channel_independence", False)),
            moving_avg=params.get("moving_avg", 25),
            embed=params.get("embed", "timeF"),
            top_k=params.get("top_k", 5),
            dropout=params.get("dropout", 0.0),
            freq=params.get("freq", "h"),
            use_norm=bool(params.get("use_norm", True)),
            decomp_method=params.get("decomp_method", "moving_avg"),
        ),
        ModelParameterConfig,
    )
