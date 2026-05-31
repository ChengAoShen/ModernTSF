"""Model registration for STID."""

from benchmark.registry import MODEL_REGISTRY
from models.stid.model import Model
from models.stid.schema import ModelParameterConfig


def register() -> None:
    """Register the STID model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STID",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            cov_dim=params.get("cov_dim", 2),
            hid_dim=params.get("hid_dim", 64),
            num_layers=params.get("num_layers", 3),
            time_of_day_size=params.get("time_of_day_size", 24),
            day_of_week_size=params.get("day_of_week_size", 7),
        ),
        ModelParameterConfig,
    )
