"""Model specification for DSTAGNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dstagnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DSTAGNN parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    d_k: int = 8
    d_v: int = 8
    n_heads: int = 4


def build_model(cfg, params):
    """Construct DSTAGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), d_model=params.get('d_model', 64), d_k=params.get('d_k', 8), d_v=params.get('d_v', 8), n_heads=params.get('n_heads', 4))
    )


SPEC = ModelSpec(
    name='DSTAGNN',
    module='models.dstagnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting',
        venue='ICML 2022',
        year=2022,
        url='https://proceedings.mlr.press/v162/lan22a.html',
    ),
    source=SourceRef(
        url='https://github.com/SYLan2019/DSTAGNN',
        revision='10da0e08ec3cf8845841741b8434fd76fd48ff84',
        license='',
    ),
    evidence="unverified",
    config_path='configs/models/DSTAGNN.toml',
    model_card='src/models/dstagnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('graph_utils',),
    deviations=(
        'The in-tree implementation was consolidated from the CauAir baseline rather than directly ported from the pinned official repository.',
        'The paper pattern-aware adjacency and temporal-distance matrix are not constructed; the adapter reuses the supplied static adjacency for both Chebyshev convolution and attention bias.',
        'Residual attention scores are not accumulated in the local scaled-dot-product attention despite being threaded between blocks.',
        'The adapter consumes only the target value channel and ignores historical and future covariates.',
        'Official initialization, preprocessing, loss, training schedule, and numerical parity are not reproduced.',
        'The official repository has no declared license file at the pinned revision.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
