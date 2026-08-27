"""Model specification for Linear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.linear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False


def build_model(cfg, params):
    """Construct Linear from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, individual=bool(params.get('individual', False)))
    )


SPEC = ModelSpec(
    name='Linear',
    module='models.linear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Are Transformers Effective for Time Series Forecasting?',
        venue='AAAI 2023',
        year=2023,
        url='https://arxiv.org/abs/2205.13504',
    ),
    source=SourceRef(
        url='https://github.com/cure-lab/LTSF-Linear',
        revision='0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/Linear.toml',
    model_card='src/models/linear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The upstream config-object constructor is replaced by the ModernTSF factory and public tensor-call wrapper.',
        'Paper-specific preprocessing, training schedules, and reported numerical results are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
