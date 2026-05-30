"""Model registration for MSGNet."""

from benchmark.registry import MODEL_REGISTRY
from models.msgnet.model import Model
from models.msgnet.schema import ModelParameterConfig


def register() -> None:
    """Register MSGNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MSGNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            c_out=params.get("c_out"),
            d_model=params.get("d_model", 128),
            d_ff=params.get("d_ff", 256),
            e_layers=params.get("e_layers", 2),
            n_heads=params.get("n_heads", 8),
            top_k=params.get("top_k", 5),
            dropout=params.get("dropout", 0.1),
            conv_channel=params.get("conv_channel", 32),
            skip_channel=params.get("skip_channel", 32),
            gcn_depth=params.get("gcn_depth", 2),
            propalpha=params.get("propalpha", 0.3),
            node_dim=params.get("node_dim", 10),
            individual=bool(params.get("individual", False)),
            embed=params.get("embed", "timeF"),
            freq=params.get("freq", "h"),
        ),
        ModelParameterConfig,
    )
