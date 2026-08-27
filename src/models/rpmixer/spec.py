"""Model specification for RPMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.rpmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated RPMixer parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None


def build_model(cfg, params):
    """Construct RPMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cov_dim=params.get('cov_dim'))
    )


SPEC = ModelSpec(
    name='RPMixer',
    module='models.rpmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/RPMixer.toml',
    model_card='src/models/rpmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
