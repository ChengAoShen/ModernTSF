"""Model specification for Autoformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
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
    config_path='configs/models/Autoformer.toml',
    model_card='src/models/autoformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('auto_correlation', 'autoformer_encdec', 'embed'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
