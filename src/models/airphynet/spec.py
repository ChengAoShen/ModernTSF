"""Model specification for AirPhyNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.airphynet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirPhyNet parameters supplied via ``model.params``."""

    enc_in: int
    latent_dim: int = 4
    rnn_units: int = 64
    ode_method: str = "dopri5"
    cov_dim: int = 2


def build_model(cfg, params):
    """Construct AirPhyNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), latent_dim=params.get('latent_dim', 4), rnn_units=params.get('rnn_units', 64), ode_method=params.get('ode_method', 'dopri5'))
    )


SPEC = ModelSpec(
    name='AirPhyNet',
    module='models.airphynet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/AirPhyNet.toml',
    model_card='src/models/airphynet/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
