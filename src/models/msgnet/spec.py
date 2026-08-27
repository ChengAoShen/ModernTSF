"""Model specification for MSGNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.msgnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 2
    n_heads: int = 8
    top_k: int = 5
    dropout: float = 0.1
    conv_channel: int = 32
    skip_channel: int = 32
    gcn_depth: int = 2
    propalpha: float = 0.3
    node_dim: int = 10
    individual: bool = False
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct MSGNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out'), d_model=params.get('d_model', 128), d_ff=params.get('d_ff', 256), e_layers=params.get('e_layers', 2), n_heads=params.get('n_heads', 8), top_k=params.get('top_k', 5), dropout=params.get('dropout', 0.1), conv_channel=params.get('conv_channel', 32), skip_channel=params.get('skip_channel', 32), gcn_depth=params.get('gcn_depth', 2), propalpha=params.get('propalpha', 0.3), node_dim=params.get('node_dim', 10), individual=bool(params.get('individual', False)), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='MSGNet',
    module='models.msgnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting',
        venue='AAAI 2024',
        year=2024,
        url='https://arxiv.org/abs/2401.00423',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/MSGNet.toml',
    model_card='src/models/msgnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'masking'),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
