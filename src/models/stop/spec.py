"""Model specification for STOP."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.stop.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STOP parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 16
    prompt_dim: int = 16
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    core: int = 4
    head: int = 4


def build_model(cfg, params):
    """Construct STOP from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], model_dim=params.get('model_dim', 16), prompt_dim=params.get('prompt_dim', 16), num_layer=params.get('num_layer', 2), hid_dim=params.get('hid_dim', 64), tod_size=params.get('tod_size', 24), kernel_size=params.get('kernel_size', 3), core=params.get('core', 4), head=params.get('head', 4))
    )


SPEC = ModelSpec(
    name='STOP',
    module='models.stop',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Robust Spatio-Temporal Centralized Interaction for OOD Learning',
        venue='ICML 2025',
        year=2025,
        url='https://proceedings.mlr.press/v267/ma25s.html',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/STOP',
        revision='8babb610ece36a4215b2f66e1ef4a154f0c4f440',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/STOP.toml',
    model_card='src/models/stop/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('base', 'marks', 'series_decomposition'),
    deviations=(
        'The core is traced to LargeST/src/models/stop.py at the pinned author revision; the BaseModel import, equivalent shared edge-padded series decomposition, and device-safe calendar-index casts were adapted locally.',
        'The public adapter fixes extra_type=1 and same=0, using a detached copy of the base predictor for residual correction as in the supplied LargeST path.',
        'Only the forecasting architecture is exposed; the paper message-perturbation environments, spatiotemporal distributionally robust optimization, OOD splits, and training losses are not reproduced by the generic runner.',
        'Raw calendar marks are converted to the normalized time-of-day and day-of-week channels expected by the source implementation.',
        'The pinned author repository contains no license file or other explicit code-license grant.',
        'No official checkpoint or numerical-parity comparison is available.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
