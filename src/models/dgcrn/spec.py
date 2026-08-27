"""Model specification for DGCRN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dgcrn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    gcn_depth: int = 1
    rnn_size: int = 16
    node_dim: int = 8
    hyper_gnn_dim: int = 8
    middle_dim: int = 2
    tanhalpha: float = 3.0
    dropout: float = 0.3


def build_model(cfg, params):
    """Construct DGCRN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), gcn_depth=params.get('gcn_depth', 1), rnn_size=params.get('rnn_size', 16), node_dim=params.get('node_dim', 8), hyper_gnn_dim=params.get('hyper_gnn_dim', 8), middle_dim=params.get('middle_dim', 2), tanhalpha=params.get('tanhalpha', 3.0), dropout=params.get('dropout', 0.3))
    )


SPEC = ModelSpec(
    name='DGCRN',
    module='models.dgcrn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution',
        venue='ACM TKDD 2023',
        year=2023,
        url='https://doi.org/10.1145/3532611',
    ),
    source=SourceRef(
        url='https://github.com/tsinghua-fib-lab/Traffic-Benchmark',
        revision='b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228',
        license='MIT',
    ),
    evidence="unverified",
    config_path='configs/models/DGCRN.toml',
    model_card='src/models/dgcrn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'The in-tree implementation is adapted from BasicTS rather than directly ported from the pinned official repository; the exact BasicTS source revision was not recorded.',
        'The preset substantially reduces recurrent, node-embedding, and hyper-network dimensions relative to the official defaults.',
        'Future time-of-day marks are used when supplied, but future target teacher forcing and the official curriculum task-level schedule are not reproduced.',
        'When no dataset graph is supplied the adapter uses identity forward and reverse supports.',
        'Official data preprocessing, masked objective, optimization schedule, and numerical parity are not recorded.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
