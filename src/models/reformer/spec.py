"""Model specification for Reformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.reformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "h"
    bucket_size: int = 4
    n_hashes: int = 4


def build_model(cfg, params):
    """Construct Reformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], c_out=params.get('c_out'), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'gelu'), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'), bucket_size=params.get('bucket_size', 4), n_hashes=params.get('n_hashes', 4))
    )


SPEC = ModelSpec(
    name='Reformer',
    module='models.reformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/Reformer.toml',
    model_card='src/models/reformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'transformer_encdec'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
