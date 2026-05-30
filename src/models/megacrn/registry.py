"""Model registration for MegaCRN."""

from benchmark.registry import MODEL_REGISTRY
from models.megacrn.model import Model
from models.megacrn.schema import ModelParameterConfig


def register() -> None:
    """Register the MegaCRN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MegaCRN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            rnn_units=params.get("rnn_units", 32),
            num_layers=params.get("num_layers", 1),
            cheb_k=params.get("cheb_k", 3),
            mem_num=params.get("mem_num", 8),
            mem_dim=params.get("mem_dim", 16),
            use_curriculum_learning=params.get("use_curriculum_learning", True),
        ),
        ModelParameterConfig,
    )
