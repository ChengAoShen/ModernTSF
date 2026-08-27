"""Model specification for PAttn."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.pattn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    activation: str = "gelu"


def build_model(cfg, params):
    """Construct PAttn from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), d_ff=params.get('d_ff', 256), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'gelu'))
    )


SPEC = ModelSpec(
    name='PAttn',
    module='models.pattn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Are Language Models Actually Useful for Time Series Forecasting?',
        venue='NeurIPS 2024',
        year=2024,
        url='https://arxiv.org/abs/2406.16964',
    ),
    source=SourceRef(url='https://github.com/thuml/Time-Series-Library', revision='4e938a1767106324dd753b2a44832bf870a0252e', license='MIT'),
    evidence="adaptation",
    config_path='configs/models/PAttn.toml',
    model_card='src/models/pattn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('self_attention_family',),
    deviations=(
        'The pad/unfold patching, per-channel patch projection, single self-attention encoder layer, normalization, and flattened forecast head match models/PAttn.py in the pinned MIT THUML source.',
        'The paper-author repository https://github.com/BennyTMT/LLMsForTimeSeries was also checked at revision 23bb8d5aa0b214056c4472e325c2d7977c1572ef but has no top-level license; it is recorded as corroborating rather than vendoring provenance.',
        'ModernTSF reuses shared attention components and removes enc_in and FullAttention factor because neither can affect this channel-independent forecast path.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
