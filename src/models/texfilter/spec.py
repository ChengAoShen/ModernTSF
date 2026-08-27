"""Model specification for TexFilter."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.texfilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    embed_size: int = 128
    hidden_size: int = 256
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct TexFilter from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], embed_size=params.get('embed_size', 128), hidden_size=params.get('hidden_size', 256), dropout=params.get('dropout', 0.0))
    )


SPEC = ModelSpec(
    name='TexFilter',
    module='models.texfilter',
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
    config_path='configs/models/TexFilter.toml',
    model_card='src/models/texfilter/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'The upstream RevIN implementation is replaced by the cataloged shared RevIN component with the same configured affine and subtract-last behavior.',
        'The unused upstream token convolution is omitted because it is not part of the TexFilter forward graph.',
        'The public wrapper omits unused mark, decoder, and mask arguments while preserving the contextual frequency-filter computation.',
        'Official preprocessing, training schedules, and reported numerical results are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
