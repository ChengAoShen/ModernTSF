"""Model specification for FITS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.fits.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False
    cut_freq: int = 24


def build_model(cfg, params):
    """Construct FITS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], individual=bool(params.get('individual', False)), cut_freq=params.get('cut_freq', 24))
    )


SPEC = ModelSpec(
    name='FITS',
    module='models.fits',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='FITS: Modeling Time Series with 10k Parameters',
        venue='ICLR 2024',
        year=2024,
        url='https://arxiv.org/abs/2307.03756',
    ),
    source=SourceRef(
        url='https://github.com/VEWOXIC/FITS',
        revision='d040bb015b6299da26d879b90dd19c80fb72c160',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/FITS.toml',
    model_card='src/models/fits/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The public wrapper returns only the final forecast slice instead of the upstream prediction-plus-low-frequency auxiliary tuple.',
        'The model package covers forecasting only; upstream anomaly-detection paths and paper experiment protocols are outside this entry.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
