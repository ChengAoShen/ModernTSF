"""Model registration for STWave."""

from benchmark.registry import MODEL_REGISTRY
from models.stwave.model import Model
from models.stwave.schema import ModelParameterConfig


def register() -> None:
    """Register the STWave model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "STWave",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            hidden_size=params.get("hidden_size", 16),
            layers=params.get("layers", 1),
            log_samples=params.get("log_samples", 1),
            time_in_day_size=params.get("time_in_day_size", 24),
            day_in_week_size=params.get("day_in_week_size", 7),
            wave_type=params.get("wave_type", "sym2"),
            wave_levels=params.get("wave_levels", 1),
        ),
        ModelParameterConfig,
    )
