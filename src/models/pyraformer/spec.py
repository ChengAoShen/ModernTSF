"""Model specification for Pyraformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.pyraformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    window_size: list[int] = [4, 4]
    inner_size: int = 5
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct Pyraformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), window_size=params.get('window_size', [4, 4]), inner_size=params.get('inner_size', 5), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='Pyraformer',
    module='models.pyraformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/Pyraformer.toml',
    model_card='src/models/pyraformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
