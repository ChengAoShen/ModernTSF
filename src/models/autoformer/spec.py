"""Model specification for Autoformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.autoformer.model import Model

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
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"


def build_model(cfg, params):
    """Construct Autoformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, label_len=cfg.task.label_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], dec_in=params['dec_in'], c_out=params['c_out'], d_model=params.get('d_model', 512), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 1), d_ff=params.get('d_ff', 2048), moving_avg=params.get('moving_avg', 25), factor=params.get('factor', 1), freq=params.get('freq', 'h'), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), activation=params.get('activation', 'gelu'))
    )


SPEC = ModelSpec(
    name='Autoformer',
    module='models.autoformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting',
        venue='NeurIPS 2021',
        year=2021,
        url='https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html',
    ),
    source=SourceRef(
        url='https://github.com/thuml/Autoformer',
        revision='51c7d416ae120b805fd5beef2f4ccf7de496a6ff',
        license='MIT',
    ),
    evidence="adaptation",
    config_path='configs/models/Autoformer.toml',
    model_card='src/models/autoformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('auto_correlation', 'autoformer_encdec', 'embed'),
    deviations=(
        'The forecast-only model and shared decomposition, Auto-Correlation, encoder/decoder, and embedding components were checked against the pinned THUML repository and THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e.',
        'Non-forecast tasks from Time-Series-Library are omitted and the generic benchmark runner owns loss, optimizer, preprocessing, and evaluation.',
        'The shared time-feature embedding consumes the benchmark six-column raw calendar representation for every frequency instead of the upstream frequency-specific normalized feature widths.',
        'label_len=0 is handled explicitly for the benchmark contract; the paper and official long-term scripts normally seed the decoder with half of the input sequence.',
        'The shared Auto-Correlation inverse FFT omits the explicit output length used upstream, so odd sequence lengths are not numerically identical; the checked contract length is even.',
        'No official checkpoint parity or reproduction of published metrics is claimed.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
