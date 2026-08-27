"""Model specification for TimeBridge."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timebridge.model import Model

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    num_p: Optional[int] = None
    ia_layers: int = 2
    pd_layers: int = 1
    ca_layers: int = 2
    stable_len: int = 3
    d_model: int = 16
    n_heads: int = 4
    d_ff: int = 128
    attn_dropout: float = 0.15
    dropout: float = 0.0
    activation: str = "gelu"
    revin: bool = True
    time_feat_dim: int = 6


def build_model(cfg, params):
    """Construct TimeBridge from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period=params.get('period', 24), num_p=params.get('num_p'), ia_layers=params.get('ia_layers', 2), pd_layers=params.get('pd_layers', 1), ca_layers=params.get('ca_layers', 2), stable_len=params.get('stable_len', 3), d_model=params.get('d_model', 16), n_heads=params.get('n_heads', 4), d_ff=params.get('d_ff', 128), attn_dropout=params.get('attn_dropout', 0.15), dropout=params.get('dropout', 0.0), activation=params.get('activation', 'gelu'), revin=bool(params.get('revin', True)), time_feat_dim=params.get('time_feat_dim', 6))
    )


SPEC = ModelSpec(
    name='TimeBridge',
    module='models.timebridge',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting',
        venue='ICML 2025',
        year=2025,
        url='https://arxiv.org/abs/2410.04442',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/TimeBridge.toml',
    model_card='src/models/timebridge/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
