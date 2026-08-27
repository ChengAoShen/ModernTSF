"""Model specification for UMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.umixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 64
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct UMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out') or params['enc_in'], d_model=params.get('d_model', 64), e_layers=params.get('e_layers', 2), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='UMixer',
    module='models.umixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting',
        venue='AAAI 2024',
        year=2024,
        url='https://arxiv.org/abs/2401.02236',
    ),
    source=SourceRef(url='https://github.com/XiangMa-Shaun/U-Mixer', revision='4192e68b85c3f11b2e19c7084f862580d97a0a55', license=''),
    evidence="unverified",
    config_path='configs/models/UMixer.toml',
    model_card='src/models/umixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'flatten_forecast_head', 'revin'),
    deviations=('Patch embedding, U-shaped temporal/channel mixing, RevIN, and Fourier stationarity correction follow the author repository.', 'Hard-coded CUDA allocation is removed and only the forecasting path is retained.', 'The author repository has no license file at the pinned revision; this provenance blocker keeps the model unverified.'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
