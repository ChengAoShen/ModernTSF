"""Model specification for MICN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.micn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int = 0  # 0 -> defaults to enc_in in the registry factory
    d_model: int = 64
    d_layers: int = 1
    dropout: float = 0.05
    embed: str = "timeF"
    freq: str = "h"
    conv_kernel: list[int] = [12, 16]


def build_model(cfg, params):
    """Construct MICN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out') or params['enc_in'], d_model=params.get('d_model', 64), d_layers=params.get('d_layers', 1), dropout=params.get('dropout', 0.05), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'), conv_kernel=params.get('conv_kernel', [12, 16]))
    )


SPEC = ModelSpec(
    name='MICN',
    module='models.micn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting',
        venue='ICLR 2023',
        year=2023,
        url='',
    ),
    source=SourceRef(url='https://github.com/wanghq21/MICN', revision='370c69b841d72246556ca05dd23163c560c22b5a', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/MICN.toml',
    model_card='src/models/micn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('autoformer_encdec', 'embed'),
    deviations=(
        'The local forecast path was adapted through THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e and structurally compared with the pinned author repository.',
        'Multi-scale hybrid decomposition, downsampling convolution, isometric convolution, transposed-convolution reconstruction, branch merge, and linear trend prediction are retained.',
        'Hardcoded CUDA placement was removed and label_len=0 falls back to value-plus-position decoder embedding without temporal marks.',
        'The pinned author repository contains no explicit code license, and training/checkpoint numerical parity is unavailable.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
