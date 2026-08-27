"""Model specification for STNorm."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.stnorm.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STNorm parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    channels: int = 16
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    tnorm_bool: bool = True
    snorm_bool: bool = True


def build_model(cfg, params):
    """Construct STNorm from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), channels=params.get('channels', 16), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2), tnorm_bool=params.get('tnorm_bool', True), snorm_bool=params.get('snorm_bool', True))
    )


SPEC = ModelSpec(
    name='STNorm',
    module='models.stnorm',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting',
        venue='KDD 2021',
        year=2021,
        url='https://doi.org/10.1145/3447548.3467330',
    ),
    source=SourceRef(url='https://github.com/GestaltCogTeam/BasicTS', revision='c218c07b6ce5e4cf908b147fd180c486346fed9c', license='Apache-2.0'),
    evidence="upstream-port",
    config_path='configs/models/STNorm.toml',
    model_card='src/models/stnorm/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=('Spatial and temporal normalization modules on the WaveNet backbone are retained.', 'Common marks are appended as BasicTS history channels although the value channel is the forecasting target.', 'The common runner and reduced dilation stack replace official training settings.', 'The terminal residual projection is omitted because the prediction head consumes the skip path and never observes that state.'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
