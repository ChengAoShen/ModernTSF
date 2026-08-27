"""Model specification for PM25_GNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.pm25gnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PM25_GNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64


def build_model(cfg, params):
    """Construct PM25_GNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), hid_dim=params.get('hid_dim', 64))
    )


SPEC = ModelSpec(
    name='PM25_GNN',
    module='models.pm25gnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting',
        venue='ACM SIGSPATIAL 2020',
        year=2020,
        url='https://doi.org/10.1145/3397536.3422208',
    ),
    source=SourceRef(
        url='https://github.com/shuowang-ai/PM2.5-GNN',
        revision='471fc60775f80492f4f224203d172868bc6eebac',
        license='MIT',
    ),
    evidence="adaptation",
    config_path='configs/models/PM25_GNN.toml',
    model_card='src/models/pm25gnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    deviations=(
        'Uses pure-PyTorch scatter aggregation and dynamic batch/device handling instead of torch_scatter and fixed runtime state.',
        'Repository adjacency weights replace the official geographic distance/direction attributes, so wind-conditioned transport edge weights are omitted.',
        'Shared calendar covariates replace the paper meteorological feature set; future covariates are still consumed autoregressively.',
        'The common runner objective and generic graph do not reproduce the KnowAir preprocessing, training loss, or evaluation protocol.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
