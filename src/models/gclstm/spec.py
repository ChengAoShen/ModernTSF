"""Model specification for GCLSTM."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.gclstm.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GCLSTM parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    Ks: int = 2


def build_model(cfg, params):
    """Construct GCLSTM from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), Ks=params.get('Ks', 2))
    )


SPEC = ModelSpec(
    name='GCLSTM',
    module='models.gclstm',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory',
        venue='Science of the Total Environment 2019',
        year=2019,
        url='https://doi.org/10.1016/j.scitotenv.2019.01.333',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/CauAir',
        revision='73dae00ca6ad14abb15174a0a0286d500e868b94',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/GCLSTM.toml',
    model_card='src/models/gclstm/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('graph_utils', 'marks'),
    deviations=(
        'No author-released implementation was identified; the immediate CauAir source declares no code license.',
        'Implements a Chebyshev graph convolution followed by a custom LSTM cell, but has not been numerically compared with the paper model.',
        'Uses normalized calendar covariates and a scaled-Laplacian fallback graph instead of the paper air-quality, meteorological, spatial, and temporal feature pipeline.',
        'Uses a direct multi-horizon decoder and the repository runner objective; the published 72-hour training and evaluation protocol is not reproduced.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
