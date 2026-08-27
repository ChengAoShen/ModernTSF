"""Model specification for TiRex."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.tirex.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    period: int = 24
    num_prompts: int = 4
    use_revin: bool = True
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    """Construct TiRex from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1), period=params.get('period', 24), num_prompts=params.get('num_prompts', 4), use_revin=bool(params.get('use_revin', True)), quantile_levels=params.get('quantile_levels'))
    )


SPEC = ModelSpec(
    name='TiRex',
    module='models.tirex',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TiRex.toml',
    model_card='src/models/tirex/README.md',
    smoke_config=None,
    capabilities=frozenset(['quantile-output', 'time-series']),
    adapter='recent-tsf',
    components=('quantile_head',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
