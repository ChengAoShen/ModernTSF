"""Model specification for Transformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.transformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int | None = None
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct Transformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], dec_in=params.get('dec_in'), c_out=params.get('c_out'), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 1), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'gelu'), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='Transformer',
    module='models.transformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Attention Is All You Need',
        venue='NeurIPS 2017',
        year=2017,
        url='https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need',
    ),
    source=SourceRef(
        url='https://github.com/thuml/Time-Series-Library',
        revision='2fb5b84ecef67c45a759f7cf82023d27afe27882',
        license='MIT',
    ),
    evidence="upstream-port",
    config_path='configs/models/Transformer.toml',
    model_card='src/models/transformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family', 'transformer_encdec'),
    deviations=(
        'This is the THUML time-series forecasting adaptation of the vanilla encoder-decoder Transformer, not the original translation pipeline.',
        'Only forecasting is retained and shared embedding, full-attention, encoder, and decoder components are reused.',
        'FullAttention ignores its compatibility factor argument, so it is no longer exposed as a model hyperparameter.',
        'The common runner constructs decoder inputs and uses forecasting datasets and losses rather than the paper translation objective.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
