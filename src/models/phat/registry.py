"""Model registration for PHAT."""

from benchmark.registry import MODEL_REGISTRY
from models.phat.model import Model
from models.phat.schema import ModelParameterConfig


def register() -> None:
    """Register the PHAT model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PHAT",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            n_heads=params.get("n_heads", 8),
            d_layers=params.get("d_layers", 1),
            attn_dropout=params.get("attn_dropout", 0.1),
            ffn_dropout=params.get("ffn_dropout", 0.1),
            ffn_expand_ratio=params.get("ffn_expand_ratio", 2.66667),
            period_topk=params.get("period_topk", 1),
            period_list=params.get("period_list"),
            ci=params.get("ci", 1),
            output_base_pred=params.get("output_base_pred", 0),
        ),
        ModelParameterConfig,
    )
