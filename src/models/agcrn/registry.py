"""Model registration for AGCRN."""

from benchmark.registry import MODEL_REGISTRY
from models.agcrn.model import Model
from models.agcrn.schema import ModelParameterConfig


def register() -> None:
    """Register the AGCRN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "AGCRN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            rnn_units=params.get("rnn_units", 32),
            embed_dim=params.get("embed_dim", 8),
            num_layers=params.get("num_layers", 1),
            cheb_k=params.get("cheb_k", 2),
            output_dim=params.get("output_dim", 1),
        ),
        ModelParameterConfig,
    )
