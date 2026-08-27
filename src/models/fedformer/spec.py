"""Model specification for FEDformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.fedformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int
    c_out: int
    freq: str = "h"
    embed: str = "timeF"
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    moving_avg: int = 25
    dropout: float = 0.1
    activation: str = "gelu"
    mode_select: str = "random"
    modes: int = 32


def build_model(cfg, params):
    """Construct FEDformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, label_len=cfg.task.label_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], dec_in=params['dec_in'], c_out=params['c_out'], d_model=params.get('d_model', 512), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 1), d_ff=params.get('d_ff', 2048), moving_avg=params.get('moving_avg', 25), freq=params.get('freq', 'h'), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), activation=params.get('activation', 'gelu'), mode_select=params.get('mode_select', 'random'), modes=params.get('modes', 32))
    )


SPEC = ModelSpec(
    name='FEDformer',
    module='models.fedformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting',
        venue='ICML 2022',
        year=2022,
        url='https://proceedings.mlr.press/v162/zhou22g.html',
    ),
    source=SourceRef(
        url='https://github.com/MAZiqing/FEDformer',
        revision='c0f6b972def125691434d62be1ecadf710ae921a',
        license='MIT',
    ),
    evidence="adaptation",
    config_path='configs/models/FEDformer.toml',
    model_card='src/models/fedformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('auto_correlation', 'autoformer_encdec', 'embed', 'fourier_correlation'),
    deviations=(
        'Only the Fourier variant is included; upstream wavelet layers are not vendored and the redundant fixed version parameter is not exposed.',
        'The forecast core and shared Fourier, decomposition, encoder/decoder, and embedding components were checked against the pinned author repository and THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e.',
        'The shared Fourier blocks use configurable attention heads and explicit real/imaginary parameters compatible with the later Time-Series-Library implementation rather than the original repository hardcoded eight-head complex tensor.',
        'The shared time-feature embedding consumes six raw calendar columns for every frequency instead of upstream frequency-specific normalized widths.',
        'label_len=0 is handled explicitly; official forecasting scripts normally use a nonzero decoder context.',
        'Training, preprocessing, objective, initialization seed, and published numerical results are not reproduced.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
