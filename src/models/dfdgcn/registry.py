"""Model registration for DFDGCN."""

from benchmark.registry import MODEL_REGISTRY
from models.dfdgcn.model import Model
from models.dfdgcn.schema import ModelParameterConfig


def register() -> None:
    """Register the DFDGCN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "DFDGCN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            num_nodes=params.get("num_nodes", params["enc_in"]),
            adj_mx=params.get("adj_mx"),
            input_dim=params.get("input_dim", 3),
            dropout=params.get("dropout", 0.3),
            residual_channels=params.get("residual_channels", 16),
            dilation_channels=params.get("dilation_channels", 16),
            skip_channels=params.get("skip_channels", 64),
            end_channels=params.get("end_channels", 128),
            kernel_size=params.get("kernel_size", 2),
            blocks=params.get("blocks", 2),
            layers=params.get("layers", 2),
            a=params.get("a", 1.0),
            fft_emb=params.get("fft_emb", 10),
            identity_emb=params.get("identity_emb", 10),
            hidden_emb=params.get("hidden_emb", 30),
            subgraph=params.get("subgraph", 20),
        ),
        ModelParameterConfig,
    )
