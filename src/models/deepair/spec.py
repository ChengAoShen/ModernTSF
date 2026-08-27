"""Model specification for DeepAir."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.deepair.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DeepAir parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64


def build_model(cfg, params):
    """Construct DeepAir from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cov_dim=params.get('cov_dim', 2), hid_dim=params.get('hid_dim', 64))
    )


SPEC = ModelSpec(
    name='DeepAir',
    module='models.deepair',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DeepAir.toml',
    model_card='src/models/deepair/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
