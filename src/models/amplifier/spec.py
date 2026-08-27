"""Model specification for Amplifier."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.amplifier.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 128
    sci: bool = False


def build_model(cfg, params):
    """Construct Amplifier from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], hidden_size=params.get('hidden_size', 128), sci=bool(params.get('sci', False)))
    )


SPEC = ModelSpec(
    name='Amplifier',
    module='models.amplifier',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting',
        venue='AAAI 2025',
        year=2025,
        url='https://arxiv.org/abs/2501.17216',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/Amplifier.toml',
    model_card='src/models/amplifier/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
