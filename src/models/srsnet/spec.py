"""Model specification for SRSNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.srsnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    patch_len: int = 24
    stride: int = 24
    hidden_size: int = 64
    dropout: float = 0.2
    head_dropout: float = 0.1
    alpha: float = 2.0
    pos: bool = True
    head_mode: str = "linear"
    affine: bool = True
    subtract_last: bool = False


def build_model(cfg, params):
    """Construct SRSNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), patch_len=params.get('patch_len', 24), stride=params.get('stride', 24), hidden_size=params.get('hidden_size', 64), dropout=params.get('dropout', 0.2), head_dropout=params.get('head_dropout', 0.1), alpha=params.get('alpha', 2.0), pos=bool(params.get('pos', True)), head_mode=params.get('head_mode', 'linear'), affine=bool(params.get('affine', True)), subtract_last=bool(params.get('subtract_last', False)))
    )


SPEC = ModelSpec(
    name='SRSNet',
    module='models.srsnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective',
        venue='NeurIPS 2025',
        year=2025,
        url='https://arxiv.org/abs/2510.14510',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/SRSNet.toml',
    model_card='src/models/srsnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'revin'),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
