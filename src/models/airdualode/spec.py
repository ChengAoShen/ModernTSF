"""Model specification for AirDualODE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
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
    paper=PaperRef(
        title='Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems',
        venue='ICLR 2025',
        year=2025,
        url='https://openreview.net/forum?id=kOJf7Dklyv',
    ),
    source=SourceRef(
        url='https://github.com/decisionintelligence/Air-DualODE',
        revision='3accfef5d3ab40f685ea29f302f76287706ba821',
        license='',
    ),
    evidence="unverified",
    config_path='configs/models/AirDualODE.toml',
    model_card='src/models/airdualode/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    deviations=(
        'The in-tree implementation was consolidated from the CauAir baseline and is not a direct port of the pinned official multi-file implementation.',
        'Its physics ODE is diffusion-only and omits the official boundary-aware advection term, wind variables, edge attributes, and learned coefficients.',
        'The generic adapter omits the official temporal-alignment contrastive loss and uses the benchmark objective only.',
        'The preset uses smaller latent dimensions and one Euler solver, while the official KnowAir preset uses distinct dopri5 and rk4 solvers with adjoint integration.',
        'When no graph is supplied the adapter uses an identity adjacency instead of the dataset graph.',
        'The official repository has no declared license file at the pinned revision.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
