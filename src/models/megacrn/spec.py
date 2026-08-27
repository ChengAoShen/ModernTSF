"""Model specification for MegaCRN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.megacrn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MegaCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are *injected* by the runner from the dataset
    (see ``src/benchmark/runner/run_one.py``) and need not be declared in TOML.
    """

    enc_in: int  # number of spatial nodes N (required)
    input_dim: int = 3
    rnn_units: int = 32
    num_layers: int = 1
    cheb_k: int = 3
    mem_num: int = 8
    mem_dim: int = 16
    use_curriculum_learning: bool = True


def build_model(cfg, params):
    """Construct MegaCRN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), rnn_units=params.get('rnn_units', 32), num_layers=params.get('num_layers', 1), cheb_k=params.get('cheb_k', 3), mem_num=params.get('mem_num', 8), mem_dim=params.get('mem_dim', 16), use_curriculum_learning=params.get('use_curriculum_learning', True))
    )


SPEC = ModelSpec(
    name='MegaCRN',
    module='models.megacrn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Spatio-Temporal Meta-Graph Learning for Traffic Forecasting',
        venue='AAAI 2023',
        year=2023,
        url='https://arxiv.org/abs/2211.14701',
    ),
    source=SourceRef(url='https://github.com/GestaltCogTeam/BasicTS', revision='c218c07b6ce5e4cf908b147fd180c486346fed9c', license='Apache-2.0'),
    evidence="adaptation",
    config_path='configs/models/MegaCRN.toml',
    model_card='src/models/megacrn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=('Memory queries, learned meta-graphs, recurrent encoder-decoder, and curriculum-learning path are retained.', 'Injected adjacency is appended as an extra normalized support although the published model learns its graph from memory.', 'ModernTSF target hooks and the common runner replace the official composite loss and training loop.'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
