"""Model specification for STDN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.stdn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STDN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (not declared in TOML). ``enc_in`` (= ``N``) is the required node count and
    is used as the ``num_nodes`` fallback.
    """

    enc_in: int
    time_slice_size: int = 60
    K: int = 4
    d: int = 8
    L: int = 1
    order: int = 2
    reference: int = 4
    out_channels: int = 1


def build_model(cfg, params):
    """Construct STDN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), time_slice_size=params.get('time_slice_size', 60), K=params.get('K', 4), d=params.get('d', 8), L=params.get('L', 1), order=params.get('order', 2), reference=params.get('reference', 4), out_channels=params.get('out_channels', 1))
    )


SPEC = ModelSpec(
    name='STDN',
    module='models.stdn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting',
        venue='AAAI 2025',
        year=2025,
        url='https://doi.org/10.1609/aaai.v39i11.33247',
    ),
    source=SourceRef(
        url='https://github.com/GestaltCogTeam/BasicTS',
        revision='c218c07b6ce5e4cf908b147fd180c486346fed9c',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/STDN.toml',
    model_card='src/models/stdn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'ModernTSF reconstructs integer day/time encodings from the shared mark contract.',
        'Dataset adjacency supplies the Laplacian positional encoding; an identity-ring fallback is used when adjacency is absent.',
        'Unused torch_geometric code and hardcoded CUDA allocations are removed without changing the active STDN path.',
        'The common runner objective and reduced display preset do not reproduce the official dataset-specific training protocol.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
