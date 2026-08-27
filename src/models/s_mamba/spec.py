"""Model specification for S_Mamba."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.s_mamba.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_state: int = 16
    d_ff: int = 128
    e_layers: int = 2
    d_conv: int = 2
    expand: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct S_Mamba from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), d_state=params.get('d_state', 16), d_ff=params.get('d_ff', 128), e_layers=params.get('e_layers', 2), d_conv=params.get('d_conv', 2), expand=params.get('expand', 1), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'gelu'), use_norm=bool(params.get('use_norm', True)), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='S_Mamba',
    module='models.s_mamba',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/S_Mamba.toml',
    model_card='src/models/s_mamba/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'mamba'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
