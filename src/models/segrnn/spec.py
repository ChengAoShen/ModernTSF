"""Model specification for SegRNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.segrnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    seg_len: int = 24


def build_model(cfg, params):
    """Construct SegRNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1), seg_len=params.get('seg_len', 24))
    )


SPEC = ModelSpec(
    name='SegRNN',
    module='models.segrnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting',
        venue='arXiv preprint',
        year=2023,
        url='https://arxiv.org/abs/2308.11200',
    ),
    source=SourceRef(url='https://github.com/lss-1138/SegRNN', revision='8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f', license='Apache-2.0'),
    evidence="upstream-port",
    config_path='configs/models/SegRNN.toml',
    model_card='src/models/segrnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=('Segment embedding, GRU recurrence, positional/channel embeddings, and parallel segment decoding match the official forecasting path.', 'The configs-object constructor is replaced by explicit parameters and the common forecast tensor contract.', 'Official dataset scripts and optimization settings are outside the model port.'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
