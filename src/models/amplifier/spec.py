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
    source=SourceRef(url='https://github.com/aikunyi/amplifier', revision='6cc089312254a0eeda7767342f690fd4536a1758', license='Apache-2.0'),
    evidence="adaptation",
    config_path='configs/models/Amplifier.toml',
    model_card='src/models/amplifier/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin', 'series_decomposition'),
    deviations=(
        'Spectrum flipping/masking energy amplification, seasonal-trend forecasting, frequency-domain energy restoration, RevIN, and optional semi-channel interaction were compared with models/Amplifier.py in the pinned Apache-2.0 author repository.',
        'ModernTSF replaces the config-object constructor, reuses shared RevIN, and conditionally constructs SCI parameters so a disabled SCI branch leaves no permanently untrained weights.',
        'The repository preset remains a compact generic benchmark configuration; checkpoint-level numerical parity is not claimed.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
