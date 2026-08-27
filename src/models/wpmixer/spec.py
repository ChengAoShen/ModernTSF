"""Model specification for WPMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.wpmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    dropout: float = 0.1
    tfactor: int = 5
    dfactor: int = 5
    wavelet: str = "db2"
    level: int = 1
    patch_len: int = 16
    stride: int = 8
    no_decomposition: bool = False


def build_model(cfg, params):
    """Construct WPMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], c_out=params.get('c_out'), d_model=params.get('d_model', 128), dropout=params.get('dropout', 0.1), tfactor=params.get('tfactor', 5), dfactor=params.get('dfactor', 5), wavelet=params.get('wavelet', 'db2'), level=params.get('level', 1), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), no_decomposition=bool(params.get('no_decomposition', False)))
    )


SPEC = ModelSpec(
    name='WPMixer',
    module='models.wpmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting',
        venue='AAAI 2025',
        year=2025,
        url='https://arxiv.org/abs/2412.17176',
    ),
    source=SourceRef(url='https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer', revision='74104c9dddd54d279eb8323f48934b4fd75fcae7', license='MIT'),
    evidence="adaptation",
    config_path='configs/models/WPMixer.toml',
    model_card='src/models/wpmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The long-term forecast path was adapted from THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e and checked against the pinned author repository.',
        'Multi-level wavelet decomposition, per-resolution patching, token/embedding mixers, resolution reconstruction, aggregation, and stationarization are retained.',
        'pytorch_wavelets/pywt were replaced by a local conv1d/conv_transpose1d zero-padding DWT/IDWT supporting only db1/haar and db2.',
        'Wavelet coefficient parity is asserted structurally for supported filters but no official checkpoint or end-to-end numerical parity is claimed.',
        'Non-forecast task paths and upstream training/data pipelines are omitted.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
