"""Model specification for DeepAir."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
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
    paper=PaperRef(
        title='Deep Distributed Fusion Network for Air Quality Prediction',
        venue='KDD 2018',
        year=2018,
        url='https://doi.org/10.1145/3219819.3219822',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/CauAir',
        revision='73dae00ca6ad14abb15174a0a0286d500e868b94',
        license='',
    ),
    evidence="unverified",
    config_path='configs/models/DeepAir.toml',
    model_card='src/models/deepair/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    deviations=(
        'No author-released reference implementation was identified; the source record points to the secondary CauAir baseline from which this entry was adapted.',
        'The paper spatial-transformation component that converts sparse station observations into pollutant-source inputs is absent.',
        'The CauAir future-side-information path was removed, so this entry does not consume weather forecasts or other known future covariates.',
        'Generic time marks replace the paper heterogeneous air-quality, meteorology, and weather-forecast feature pipeline.',
        'The CauAir repository has no declared license file at the pinned revision.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
