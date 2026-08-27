"""Model specification for Informer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.informer.model import Model

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
    factor: int = 3
    activation: str = "gelu"
    distil: bool = True
    embed: str = "timeF"
    freq: str = "h"


def build_model(cfg, params):
    """Construct Informer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], dec_in=params.get('dec_in'), c_out=params.get('c_out'), d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), e_layers=params.get('e_layers', 2), d_layers=params.get('d_layers', 1), d_ff=params.get('d_ff', 256), dropout=params.get('dropout', 0.1), factor=params.get('factor', 3), activation=params.get('activation', 'gelu'), distil=bool(params.get('distil', True)), embed=params.get('embed', 'timeF'), freq=params.get('freq', 'h'))
    )


SPEC = ModelSpec(
    name='Informer',
    module='models.informer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting',
        venue='AAAI 2021',
        year=2021,
        url='https://doi.org/10.1609/aaai.v35i12.17325',
    ),
    source=SourceRef(
        url='https://github.com/thuml/Time-Series-Library',
        revision='2fb5b84ecef67c45a759f7cf82023d27afe27882',
        license='MIT',
    ),
    evidence="upstream-port",
    config_path='configs/models/Informer.toml',
    model_card='src/models/informer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family', 'transformer_encdec'),
    deviations=(
        'Only the long-term forecasting branch is retained; imputation, anomaly detection, and classification branches are omitted.',
        'THUML embedding, ProbSparse attention, attention distillation, and encoder-decoder components are reused from shared modules.',
        'The common runner constructs decoder start tokens and future marks and supplies the repository-wide training objective.',
        'The display preset is smaller and uses label_len=0 rather than the paper experiment settings.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
