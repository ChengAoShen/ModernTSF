"""Model registration for DCRNN."""

from benchmark.registry import MODEL_REGISTRY
from models.dcrnn.model import Model
from models.dcrnn.schema import ModelParameterConfig


def register() -> None:
    """Register the DCRNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DCRNN",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            n_filters=params.get("n_filters", 64),
            max_diffusion_step=params.get("max_diffusion_step", 2),
            filter_type=params.get("filter_type", "doubletransition"),
            num_rnn_layers=params.get("num_rnn_layers", 2),
        ),
        ModelParameterConfig,
    )
