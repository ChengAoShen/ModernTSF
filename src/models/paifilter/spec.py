"""Model specification for PaiFilter."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.paifilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 256


def build_model(cfg, params):
    """Construct PaiFilter from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], hidden_size=params.get('hidden_size', 256))
    )


SPEC = ModelSpec(
    name='PaiFilter',
    module='models.paifilter',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='FilterNet: Harnessing Frequency Filters for Time Series Forecasting',
        venue='NeurIPS 2024',
        year=2024,
        url='https://arxiv.org/abs/2411.01623',
    ),
    source=SourceRef(
        url='https://github.com/aikunyi/FilterNet',
        revision='cdb321c4e338e0c07b45cee92f54b3c5bd5a809e',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/PaiFilter.toml',
    model_card='src/models/paifilter/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'The upstream RevIN implementation is replaced by the cataloged shared RevIN component with the same configured affine and subtract-last behavior.',
        'The public wrapper omits unused mark, decoder, and mask arguments while preserving the plain circular-convolution filter computation.',
        'Official preprocessing, training schedules, and reported numerical results are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
