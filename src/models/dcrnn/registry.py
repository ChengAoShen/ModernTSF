"""Model registration for DCRNN."""

from benchmark.registry import MODEL_REGISTRY
from models.dcrnn.model import Model
from models.dcrnn.schema import ModelParameterConfig


def register() -> None:
    """Register the DCRNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DCRNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            rnn_units=params.get("rnn_units", 16),
            num_rnn_layers=params.get("num_rnn_layers", 1),
            max_diffusion_step=params.get("max_diffusion_step", 2),
            cl_decay_steps=params.get("cl_decay_steps", 2000),
            use_curriculum_learning=params.get("use_curriculum_learning", False),
        ),
        ModelParameterConfig,
    )
