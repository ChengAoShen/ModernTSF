"""Model registration for QuantilePatchTST."""

from benchmark.registry import MODEL_REGISTRY
from models.quantile_patchtst.model import Model
from models.quantile_patchtst.schema import ModelParameterConfig


def register() -> None:
    """Register QuantilePatchTST model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "QuantilePatchTST",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            features=cfg.task.features,
            patch_len=params.get("patch_len", 16),
            stride=params.get("stride", 8),
            padding_patch=params.get("padding_patch", "end"),
            e_layers=params.get("e_layers", 3),
            d_model=params.get("d_model", 512),
            n_heads=params.get("n_heads", 8),
            d_k=params.get("d_k"),
            d_v=params.get("d_v"),
            d_ff=params.get("d_ff", 2048),
            activation=params.get("activation", "gelu"),
            norm=params.get("norm", "BatchNorm"),
            attn_dropout=params.get("attn_dropout", 0.0),
            res_dropout=params.get("res_dropout", 0.0),
            ffn_dropout=params.get("ffn_dropout", 0.0),
            proj_dropout=params.get("proj_dropout", 0.0),
            head_dropout=params.get("head_dropout", 0.0),
            pre_norm=bool(params.get("pre_norm", False)),
            pe=params.get("pe", "zeros"),
            learn_pe=bool(params.get("learn_pe", False)),
            head_type=params.get("head_type", "flatten"),
            individual=bool(params.get("individual", False)),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", False)),
            subtract_last=bool(params.get("subtract_last", False)),
            quantile_levels=params.get("quantile_levels"),
        ),
        ModelParameterConfig,
    )
