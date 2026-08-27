"""Model specification for AirDualODE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.airdualode.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirDualODE parameters supplied via ``model.params``."""

    enc_in: int
    phy_latent_dim: int = 16
    unk_latent_dim: int = 16
    gcn_hidden_dim: int = 32
    n_heads: int = 4
    ode_method: str = "euler"
    cov_dim: int = 2


def build_model(cfg, params):
    """Construct AirDualODE from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), phy_latent_dim=params.get('phy_latent_dim', 16), unk_latent_dim=params.get('unk_latent_dim', 16), gcn_hidden_dim=params.get('gcn_hidden_dim', 32), n_heads=params.get('n_heads', 4), ode_method=params.get('ode_method', 'euler'))
    )


SPEC = ModelSpec(
    name='AirDualODE',
    module='models.airdualode',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/AirDualODE.toml',
    model_card='src/models/airdualode/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
