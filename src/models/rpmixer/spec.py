"""Model specification for RPMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
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
    paper=PaperRef(
        title='RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data',
        venue='KDD 2024',
        year=2024,
        url='https://doi.org/10.1145/3637528.3671881',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/CauAir',
        revision='73dae00ca6ad14abb15174a0a0286d500e868b94',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/RPMixer.toml',
    model_card='src/models/rpmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'The paper does not provide an official code release; the immediate CauAir source declares no code license.',
        'The local all-MLP stack contains fixed random projections, frequency-domain mixing, residual blocks, and reversible normalization, but numerical parity is unverified.',
        'Shared calendar covariates are flattened with values instead of the paper benchmark-specific feature construction.',
        'The common runner objective and generic eight-layer defaults do not reproduce the published dataset and optimization protocol.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
