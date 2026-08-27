"""Model specification for MAGE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.mage.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MAGE parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 64
    recur_num: int = 8
    topk: int = 2
    node_dim: int = 16
    tod_size: int = 24


def build_model(cfg, params):
    """Construct MAGE from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], model_dim=params.get('model_dim', 64), recur_num=params.get('recur_num', 8), topk=params.get('topk', 2), node_dim=params.get('node_dim', 16), tod_size=params.get('tod_size', 24))
    )


SPEC = ModelSpec(
    name='MAGE',
    module='models.mage',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting',
        venue='NeurIPS 2025',
        year=2025,
        url='https://proceedings.neurips.cc/paper_files/paper/2025/hash/54c9bfb0885ae07f23607f617ab64c2b-Abstract-Conference.html',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/MAGE',
        revision='f1fdd27da4e72a140c4f341f94d368fbcaec7507',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/MAGE.toml',
    model_card='src/models/mage/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('base', 'marks'),
    deviations=(
        'The model core matches src/models/MAGE.py at the pinned author revision except for the BaseModel import path.',
        'The upstream architecture fixes its three transformer depths to 1, 2, and 3; the previously exposed blocknum parameter was inert and has been removed.',
        'The adapter derives time-of-day and day-of-week inputs from the shared raw calendar marks and updates the upstream fixed batch-size field for partial batches.',
        'The training-only expert-usage counts returned upstream are discarded by the forecasting interface.',
        'The pinned author repository contains no license file or other explicit code-license grant.',
        'No official checkpoint or numerical-parity comparison is available.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
