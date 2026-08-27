"""Model specification for TimeKAN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timekan.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 16
    e_layers: int = 1
    down_sampling_window: int = 2
    down_sampling_layers: int = 1
    begin_order: int = 0
    moving_avg: int = 25
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    use_norm: int = 1


def build_model(cfg, params):
    """Construct TimeKAN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out', None), d_model=params.get('d_model', 16), e_layers=params.get('e_layers', 1), down_sampling_window=params.get('down_sampling_window', 2), down_sampling_layers=params.get('down_sampling_layers', 1), begin_order=params.get('begin_order', 0), moving_avg=params.get('moving_avg', 25), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'), use_norm=params.get('use_norm', 1))
    )


SPEC = ModelSpec(
    name='TimeKAN',
    module='models.timekan',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting',
        venue='arXiv preprint',
        year=2025,
        url='https://arxiv.org/abs/2502.06910',
    ),
    source=SourceRef(url='https://github.com/huangst21/TimeKAN', revision='3a7c366a9e8547fd8840c5d27f25ee3e30615e33', license='Apache-2.0'),
    evidence="upstream-port",
    config_path='configs/models/TimeKAN.toml',
    model_card='src/models/timekan/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('autoformer_encdec', 'embed', 'revin'),
    deviations=('Chebyshev KAN layers, frequency decomposition, multiscale KAN mixing, and residual frequency reconstruction match the official forecast path.', 'The configs-object constructor and local utility imports are replaced by explicit parameters and shared components.', 'The unused calendar embedding is removed because the official TimeKAN forecast path always embeds each channel with x_mark=None.'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
