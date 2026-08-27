"""Model specification for MambaSimple."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.mambasimple.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    d_ff: int = 16
    e_layers: int = 2
    expand: int = 2
    d_conv: int = 4
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct MambaSimple from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out'), d_model=params.get('d_model', 128), d_ff=params.get('d_ff', 16), e_layers=params.get('e_layers', 2), expand=params.get('expand', 2), d_conv=params.get('d_conv', 4), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='MambaSimple',
    module='models.mambasimple',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MambaSimple.toml',
    model_card='src/models/mambasimple/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'mamba'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
