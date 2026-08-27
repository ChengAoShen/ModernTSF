"""Model specification for MoFo."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.mofo.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MoFo parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    periodic: int = 24
    head: int = 4
    d_layers: int = 1
    bias: int = 1
    cias: int = 1


def build_model(cfg, params):
    """Construct MoFo from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), periodic=params.get('periodic', 24), head=params.get('head', 4), d_layers=params.get('d_layers', 1), bias=params.get('bias', 1), cias=params.get('cias', 1))
    )


SPEC = ModelSpec(
    name='MoFo',
    module='models.mofo',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling',
        venue='NeurIPS 2025',
        year=2025,
        url='https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/MoFo',
        revision='2d14b47ea839c3809952b412340d72393f2521dc',
        license='MIT',
    ),
    evidence="upstream-port",
    config_path='configs/models/MoFo.toml',
    model_card='src/models/mofo/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('marks',),
    deviations=(
        'The local core retains the official forecast path but removes unrelated classification, anomaly-detection, and imputation task branches.',
        'Unused upstream layers are instantiated transiently to preserve parameter initialization order but are not stored in the forecast-only module.',
        'The adapter reconstructs the TFB-normalized hour, minute, and weekday columns from the benchmark raw calendar marks.',
        'Only periodic values 24, 96, 144, and 288 are supported by the upstream calendar-position logic.',
        'The generic benchmark runner supplies its own data pipeline, objective, optimizer, and evaluation rather than reproducing the upstream experiment scripts.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
