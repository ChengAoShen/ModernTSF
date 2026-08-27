"""Model specification for PHAT."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.phat.model import Model

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PHAT parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    n_heads: int = 8
    d_layers: int = 1
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    ffn_expand_ratio: float = 2.66667
    period_topk: int = 1
    period_list: Optional[list[int]] = None
    ci: int = 1
    output_base_pred: int = 0


def build_model(cfg, params):
    """Construct PHAT from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), n_heads=params.get('n_heads', 8), d_layers=params.get('d_layers', 1), attn_dropout=params.get('attn_dropout', 0.1), ffn_dropout=params.get('ffn_dropout', 0.1), ffn_expand_ratio=params.get('ffn_expand_ratio', 2.66667), period_topk=params.get('period_topk', 1), period_list=params.get('period_list'), ci=params.get('ci', 1), output_base_pred=params.get('output_base_pred', 0))
    )


SPEC = ModelSpec(
    name='PHAT',
    module='models.phat',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting',
        venue='arXiv preprint',
        year=2026,
        url='https://arxiv.org/abs/2602.00654',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/PHAT.toml',
    model_card='src/models/phat/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
