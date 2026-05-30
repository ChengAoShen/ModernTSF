"""Model registration for D2STGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.d2stgnn.model import Model
from models.d2stgnn.schema import ModelParameterConfig


def register() -> None:
    """Register the D2STGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "D2STGNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            num_feat=params.get("num_feat", 1),
            num_hidden=params.get("num_hidden", 16),
            node_hidden=params.get("node_hidden", 8),
            time_emb_dim=params.get("time_emb_dim", 8),
            k_s=params.get("k_s", 2),
            k_t=params.get("k_t", 3),
            gap=params.get("gap", 1),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.1),
            time_in_day_size=params.get("time_in_day_size", 288),
            day_in_week_size=params.get("day_in_week_size", 7),
            forecast_dim=params.get("forecast_dim", 64),
            output_hidden=params.get("output_hidden", 128),
        ),
        ModelParameterConfig,
    )
