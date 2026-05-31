"""Model registration for AirDualODE."""

from benchmark.registry import MODEL_REGISTRY
from models.airdualode.model import Model
from models.airdualode.schema import ModelParameterConfig


def register() -> None:
    """Register the AirDualODE model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "AirDualODE",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            phy_latent_dim=params.get("phy_latent_dim", 16),
            unk_latent_dim=params.get("unk_latent_dim", 16),
            gcn_hidden_dim=params.get("gcn_hidden_dim", 32),
            n_heads=params.get("n_heads", 4),
            ode_method=params.get("ode_method", "euler"),
        ),
        ModelParameterConfig,
    )
