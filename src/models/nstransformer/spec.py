"""Model specification for NSTransformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.nstransformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int | None = None
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "h"
    p_hidden_dims: list[int] = [128, 128]
    p_hidden_layers: int = 2


def build_model(cfg, params):
    """Construct NSTransformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], dec_in=params.get('dec_in'), c_out=params.get('c_out'), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 1), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), factor=params.get('factor', 3), activation=params.get('activation', 'gelu'), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'), p_hidden_dims=params.get('p_hidden_dims', [128, 128]), p_hidden_layers=params.get('p_hidden_layers', 2))
    )


SPEC = ModelSpec(
    name='NSTransformer',
    module='models.nstransformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting',
        venue='NeurIPS 2022',
        year=2022,
        url='https://arxiv.org/abs/2205.14415',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/NSTransformer.toml',
    model_card='src/models/nstransformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'masking', 'self_attention_family', 'transformer_encdec'),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
