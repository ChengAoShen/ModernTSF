"""Model specification for Pyraformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.pyraformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    window_size: list[int] = [4, 4]
    inner_size: int = 5
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct Pyraformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), window_size=params.get('window_size', [4, 4]), inner_size=params.get('inner_size', 5), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='Pyraformer',
    module='models.pyraformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting',
        venue='ICLR 2022',
        year=2022,
        url='https://openreview.net/forum?id=0EXmFzUn5I',
    ),
    source=SourceRef(
        url='https://github.com/thuml/Time-Series-Library',
        revision='3a4819420d14095354aae96750ce8c499ef5f05e',
        license='MIT',
    ),
    evidence="upstream-port",
    config_path='configs/models/Pyraformer.toml',
    model_card='src/models/pyraformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family'),
    deviations=(
        'Only the THUML long-term forecasting path is retained; short forecasting and non-forecasting task branches are omitted.',
        'The pyramidal mask, convolutional scale construction, inter-scale reference gathering, and direct multi-horizon projection are retained.',
        'Shared DataEmbedding and FullAttention replace duplicate leaf implementations without changing their tensor contracts.',
        'The common runner objective and reduced preset do not reproduce the official benchmark optimization protocol.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
