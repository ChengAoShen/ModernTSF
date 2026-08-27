"""Model specification for STGODE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.stgode.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGODE parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3


def build_model(cfg, params):
    """Construct STGODE from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3))
    )


SPEC = ModelSpec(
    name='STGODE',
    module='models.stgode',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting',
        venue='KDD 2021',
        year=2021,
        url='https://doi.org/10.1145/3447548.3467430',
    ),
    source=SourceRef(
        url='https://github.com/GestaltCogTeam/BasicTS',
        revision='c218c07b6ce5e4cf908b147fd180c486346fed9c',
        license='Apache-2.0',
    ),
    evidence="adaptation",
    config_path='configs/models/STGODE.toml',
    model_card='src/models/stgode/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'ModernTSF reconstructs BasicTS history_data from the common value/mark signature.',
        'The semantic graph reuses normalized dataset adjacency instead of the paper DTW graph computed from the full training series.',
        'The torchdiffeq single-step Euler call is replaced by its algebraically equivalent explicit Euler update.',
        'A conditional-expression precedence bug that bypassed temporal convolutions when channel widths matched is corrected.',
        'The common runner objective and generic calendar inputs do not reproduce the official preprocessing and training protocol.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
