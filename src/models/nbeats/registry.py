"""Model registration for NBeats."""

from benchmark.registry import MODEL_REGISTRY
from models.nbeats.model import Model
from models.nbeats.schema import ModelParameterConfig


def register() -> None:
    """Register NBeats model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "NBeats",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            stack_types=tuple(
                params.get("stack_types", ["trend", "seasonality", "generic"])
            ),
            nb_blocks_per_stack=params.get("nb_blocks_per_stack", 3),
            thetas_dim=tuple(params.get("thetas_dim", [4, 8, 8])),
            hidden_layer_units=params.get("hidden_layer_units", 256),
            share_weights_in_stack=bool(params.get("share_weights_in_stack", False)),
            nb_harmonics=params.get("nb_harmonics", None),
        ),
        ModelParameterConfig,
    )
