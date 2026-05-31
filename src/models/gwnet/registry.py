"""Model registration for GWNet."""

from benchmark.registry import MODEL_REGISTRY
from models.gwnet.model import Model
from models.gwnet.schema import ModelParameterConfig


def register() -> None:
    """Register the GWNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "GWNet",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            adj_mx=kw.get("graph_context", {}).get("adj_mx"),
            cov_dim=params.get("cov_dim", 2),
            residual_channels=params.get("residual_channels", 32),
            dilation_channels=params.get("dilation_channels", 32),
            skip_channels=params.get("skip_channels", 64),
            end_channels=params.get("end_channels", 128),
            dropout=params.get("dropout", 0.3),
            blocks=params.get("blocks", 4),
            layers=params.get("layers", 2),
            adp_adj=params.get("adp_adj", True),
            adj_type=params.get("adj_type", "doubletransition"),
        ),
        ModelParameterConfig,
    )
