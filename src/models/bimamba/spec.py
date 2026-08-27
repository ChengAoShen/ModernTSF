"""Model specification for BiMamba."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.bimamba.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    d_state: int = 16
    e_layers: int = 2
    expand: int = 2
    d_conv: int = 4
    dropout: float = 0.1
    share_ffn: bool = False
    share_norm: bool = False
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct BiMamba from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out', None), d_model=params.get('d_model', 128), d_state=params.get('d_state', 16), e_layers=params.get('e_layers', 2), expand=params.get('expand', 2), d_conv=params.get('d_conv', 4), dropout=params.get('dropout', 0.1), share_ffn=bool(params.get('share_ffn', False)), share_norm=bool(params.get('share_norm', False)), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='BiMamba',
    module='models.bimamba',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/BiMamba.toml',
    model_card='src/models/bimamba/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'mamba'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
