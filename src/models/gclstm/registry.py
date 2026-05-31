"""Model registration for GCLSTM."""

from benchmark.registry import MODEL_REGISTRY
from models.gclstm.model import Model
from models.gclstm.schema import ModelParameterConfig


def register() -> None:
    """Register the GCLSTM model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "GCLSTM",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=params.get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            Ks=params.get("Ks", 3),
            Kt=params.get("Kt", 3),
            blocks=params.get("blocks"),
            drop_prob=params.get("drop_prob", 0.0),
        ),
        ModelParameterConfig,
    )
