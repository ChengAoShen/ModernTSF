"""Model registration for AirPhyNet."""

from benchmark.registry import MODEL_REGISTRY
from models.airphynet.model import Model
from models.airphynet.schema import ModelParameterConfig


def register() -> None:
    """Register the AirPhyNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "AirPhyNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            latent_dim=params.get("latent_dim", 4),
            rnn_units=params.get("rnn_units", 64),
            ode_method=params.get("ode_method", "dopri5"),
        ),
        ModelParameterConfig,
    )
