"""Model specification for RLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.rlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False
    affine: bool = False
    subtract_last: bool = False


def build_model(cfg, params):
    """Construct RLinear from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, individual=bool(params.get('individual', False)), affine=bool(params.get('affine', False)), subtract_last=bool(params.get('subtract_last', False)))
    )


SPEC = ModelSpec(
    name='RLinear',
    module='models.rlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/RLinear.toml',
    model_card='src/models/rlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('channel_wise_linear', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
