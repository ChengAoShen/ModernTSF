"""Model specification for TimeXer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timexer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    patch_len: int = 16
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
    use_norm: bool = True
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct TimeXer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), patch_len=params.get('patch_len', 16), dropout=params.get('dropout', 0.1), factor=params.get('factor', 3), activation=params.get('activation', 'gelu'), use_norm=bool(params.get('use_norm', True)), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='TimeXer',
    module='models.timexer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables',
        venue='NeurIPS 2024',
        year=2024,
        url='https://arxiv.org/abs/2402.19072',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/TimeXer.toml',
    model_card='src/models/timexer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family'),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
