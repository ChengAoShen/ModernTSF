"""Model specification for GAGNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.gagnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GAGNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    group_num: int = 4


def build_model(cfg, params):
    """Construct GAGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), d_model=params.get('d_model', 64), n_heads=params.get('n_heads', 4), num_layers=params.get('num_layers', 3), dropout=params.get('dropout', 0.1), group_num=params.get('group_num', 4))
    )


SPEC = ModelSpec(
    name='GAGNN',
    module='models.gagnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting',
        venue='ACM TKDD 2024',
        year=2024,
        url='https://doi.org/10.1145/3631713',
    ),
    source=SourceRef(
        url='https://github.com/Friger/GAGNN',
        revision='509ac7d6eb55914979fc45f6d23e967021cfd270',
        license='MIT',
    ),
    evidence="adaptation",
    config_path='configs/models/GAGNN.toml',
    model_card='src/models/gagnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    deviations=(
        'Replaces the official torch_geometric message passing with pure-PyTorch mean aggregation.',
        'Uses repository-provided adjacency and zero location features instead of the paper city graph, coordinates, and distance threshold.',
        'Uses calendar features from the shared input contract rather than the official month/week/hour inputs and Chinese city dataset preprocessing.',
        'Uses a direct multi-horizon projection and unit group-edge weights instead of the official decoder and learned group-correlation edge attributes.',
        'The generic preset uses four groups rather than the paper-selected fifteen and the repository runner rather than the official summed-L1 objective.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
