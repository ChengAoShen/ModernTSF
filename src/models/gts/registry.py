"""Model registration for GTS."""

from benchmark.registry import MODEL_REGISTRY
from models.gts.model import Model
from models.gts.schema import ModelParameterConfig


def register() -> None:
    """Register the GTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "GTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            rnn_units=params.get("rnn_units", 16),
            num_rnn_layers=params.get("num_rnn_layers", 1),
            max_diffusion_step=params.get("max_diffusion_step", 2),
            embedding_dim=params.get("embedding_dim", 16),
            node_feats_len=params.get("node_feats_len", 40),
            k=params.get("k", 3),
            temp=params.get("temp", 0.5),
        ),
        ModelParameterConfig,
    )
