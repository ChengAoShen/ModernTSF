"""Model specification for MQRNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.mqrnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 64
    num_layers: int = 1
    decoder_hidden: int = 64
    dropout: float = 0.1
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    """Construct MQRNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, hidden_size=params.get('hidden_size', 64), num_layers=params.get('num_layers', 1), decoder_hidden=params.get('decoder_hidden', 64), dropout=params.get('dropout', 0.1), quantile_levels=params.get('quantile_levels'))
    )


SPEC = ModelSpec(
    name='MQRNN',
    module='models.mqrnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='A Multi-Horizon Quantile Recurrent Forecaster',
        venue='NeurIPS 2017 Time Series Workshop',
        year=2017,
        url='https://arxiv.org/abs/1711.11053',
    ),
    source=SourceRef(),
    evidence="adaptation",
    config_path='configs/models/MQRNN.toml',
    model_card='src/models/mqrnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['quantile-output', 'time-series']),
    components=('quantile_head',),
    deviations=(
        'This is an in-repository paper-inspired adaptation; no author implementation or pinned upstream code has been established.',
        'It uses a shared channel-independent GRU and a joint horizon MLP instead of the paper global/local decoder with static and future covariates.',
        'The shared monotone QuantileHead enforces non-crossing quantiles, which is a ModernTSF composition rather than a paper component.',
        'Paper dataset protocol and reported numerical results remain unverified.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
