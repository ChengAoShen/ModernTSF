"""Model specification for iTransformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.itransformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    freq: str = "h"
    embed: str = "timeF"
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 2048
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    output_attention: bool = False
    use_norm: bool = True
    class_strategy: str = "projection"


def build_model(cfg, params):
    """Construct iTransformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, d_model=params.get('d_model', 512), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 2048), factor=params.get('factor', 1), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), activation=params.get('activation', 'gelu'), output_attention=bool(params.get('output_attention', False)), use_norm=bool(params.get('use_norm', True)), freq=params.get('freq', 'h'), class_strategy=params.get('class_strategy', 'projection'))
    )


SPEC = ModelSpec(
    name='iTransformer',
    module='models.itransformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='iTransformer: Inverted Transformers Are Effective for Time Series Forecasting',
        venue='ICLR 2024',
        year=2024,
        url='https://arxiv.org/abs/2310.06625',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/iTransformer.toml',
    model_card='src/models/itransformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family', 'transformer_encdec'),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
