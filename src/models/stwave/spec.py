"""Model specification for STWave."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.stwave.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STWave parameters supplied via ``model.params``.

    ``enc_in`` (the number of spatial nodes ``N``) is required. ``num_nodes``
    and ``adj_mx`` are injected by the runner from the dataset and are NOT
    declared in the TOML. ``hidden_size`` doubles as the number of Laplacian
    eigenvectors used for the spatial positional encoding, so it is clamped to
    ``N`` at construction.
    """

    enc_in: int
    input_dim: int = 3
    hidden_size: int = 16
    layers: int = 1
    log_samples: int = 1
    time_in_day_size: int = 24
    day_in_week_size: int = 7
    wave_type: str = "sym2"
    wave_levels: int = 1


def build_model(cfg, params):
    """Construct STWave from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), hidden_size=params.get('hidden_size', 16), layers=params.get('layers', 1), log_samples=params.get('log_samples', 1), time_in_day_size=params.get('time_in_day_size', 24), day_in_week_size=params.get('day_in_week_size', 7), wave_type=params.get('wave_type', 'sym2'), wave_levels=params.get('wave_levels', 1))
    )


SPEC = ModelSpec(
    name='STWave',
    module='models.stwave',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks',
        venue='ICDE 2023',
        year=2023,
        url='https://arxiv.org/abs/2112.02740',
    ),
    source=SourceRef(url='https://github.com/GestaltCogTeam/BasicTS', revision='c218c07b6ce5e4cf908b147fd180c486346fed9c', license='Apache-2.0'),
    evidence="adaptation",
    config_path='configs/models/STWave.toml',
    model_card='src/models/stwave/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=('Wavelet disentanglement, spectral graph encoding, sparse spatial attention, and temporal stacks are retained.', 'Graph eigenpairs and neighbor samples are rebuilt from injected adjacency with local numerical utilities.', 'The wrapper always uses the inference output and common loss rather than the upstream auxiliary training objective.', 'The unused auxiliary low-frequency head is omitted, and the non-differentiable top-k ranking projection is frozen.'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
