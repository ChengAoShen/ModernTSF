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
    source=SourceRef(url='https://github.com/wxie9/CARD', revision='ca6d34bcf26355bfdb6fc05f49c66e7601817f66', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/CARD.toml',
    model_card='src/models/card/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'Channel-aligned token/channel dual attention, EMA smoothing, low-rank dynamic projection, and token blending were compared with long_term_forecast_l96/models/CARD.py in the pinned author repository.',
        'ModernTSF replaces the mutable config-object interface, computes token counts locally, and exposes only the forecasting path.',
        'Branch-specific statistic/class tokens and dynamic-projection layers are constructed conditionally so inactive branches do not leave permanently untrained parameters.',
        'The paper-specific robust signal-decay loss is not selected automatically by the model and must be supplied by the experiment loss configuration.',
        'The author repository has no explicit code license and no checkpoint-level numerical parity evidence; verification therefore remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
