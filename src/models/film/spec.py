"""Model specification for FiLM."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.film.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    e_layers: int = 2
    ratio: float = 0.5
    multiscale: list[int] = [1, 2, 4]
    window_size: list[int] = [256]


def build_model(cfg, params):
    """Construct FiLM from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], e_layers=params.get('e_layers', 2), ratio=params.get('ratio', 0.5), multiscale=params.get('multiscale', [1, 2, 4]), window_size=params.get('window_size', [256]))
    )


SPEC = ModelSpec(
    name='FiLM',
    module='models.film',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting',
        venue='NeurIPS 2022',
        year=2022,
        url='https://arxiv.org/abs/2205.08897',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/FiLM.toml',
    model_card='src/models/film/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
