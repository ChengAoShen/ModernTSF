"""Model specification for ASTGCN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.astgcn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated ASTGCN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    nb_block: int = 2
    K: int = 3
    nb_chev_filter: int = 64
    nb_time_filter: int = 64


def build_model(cfg, params):
    """Construct ASTGCN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), nb_block=params.get('nb_block', 2), K=params.get('K', 3), nb_chev_filter=params.get('nb_chev_filter', 64), nb_time_filter=params.get('nb_time_filter', 64))
    )


SPEC = ModelSpec(
    name='ASTGCN',
    module='models.astgcn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting',
        venue='AAAI 2019',
        year=2019,
        url='https://doi.org/10.1609/aaai.v33i01.3301922',
    ),
    source=SourceRef(
        url='https://github.com/guoshnBJTU/ASTGCN-r-pytorch',
        revision='2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296',
        license='',
    ),
    evidence="unverified",
    config_path='configs/models/ASTGCN.toml',
    model_card='src/models/astgcn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('graph_utils', 'marks'),
    deviations=(
        'The in-tree core was adapted from the CauAir ASTGCN baseline, not copied from the pinned official repository.',
        'This entry implements one ASTGCN branch; the paper combines recent, daily-periodic, and weekly-periodic branches with learned fusion.',
        'When no graph is supplied the adapter uses a dense all-ones adjacency, which is not the paper PeMS graph preprocessing.',
        'Paper/upstream training, masked objective, and preprocessing are handled by the generic benchmark rather than reproduced here.',
        'The official repository has no declared license file at the pinned revision.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
