"""Model specification for ETSformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.etsformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 2
    d_ff: int = 256
    top_k: int = 3
    dropout: float = 0.1
    activation: str = "sigmoid"
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct ETSformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 2), d_ff=params.get('d_ff', 256), top_k=params.get('top_k', 3), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'sigmoid'), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='ETSformer',
    module='models.etsformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='ETSformer: Exponential Smoothing Transformers for Time-series Forecasting',
        venue='arXiv preprint',
        year=2022,
        url='https://arxiv.org/abs/2202.01381',
    ),
    source=SourceRef(
        url='https://github.com/thuml/Time-Series-Library',
        revision='230805fe9f451b61e34b96116d995b417e343ac0',
        license='MIT',
    ),
    evidence="upstream-port",
    config_path='configs/models/ETSformer.toml',
    model_card='src/models/etsformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed',),
    deviations=(
        'Only the long-term forecasting path is retained; non-forecasting branches are omitted.',
        'ETS exponential smoothing, growth, Fourier seasonality, damping, and level-update blocks are vendored locally while DataEmbedding is shared.',
        'Output width is fixed to enc_in because the upstream level residual requires c_out == enc_in; the incompatible override is removed.',
        'Dead feed-forward and normalization parameters in the terminal encoder layer are omitted because its residual state is never consumed.',
        'The common runner objective and reduced preset do not reproduce the official dataset-specific training protocol.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
