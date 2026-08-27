"""Model specification for CARD."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.card.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    dp_rank: int = 8
    merge_size: int = 2
    momentum: float = 0.1
    alpha: float = 0.5
    use_statistic: bool = False


def build_model(cfg, params):
    """Construct CARD from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), dp_rank=params.get('dp_rank', 8), merge_size=params.get('merge_size', 2), momentum=params.get('momentum', 0.1), alpha=params.get('alpha', 0.5), use_statistic=bool(params.get('use_statistic', False)))
    )


SPEC = ModelSpec(
    name='CARD',
    module='models.card',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting',
        venue='ICLR 2024',
        year=2024,
        url='https://arxiv.org/abs/2305.12095',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/CARD.toml',
    model_card='src/models/card/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
