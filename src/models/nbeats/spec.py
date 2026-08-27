"""Model specification for NBeats."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.nbeats.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    stack_types: list[str] = ["trend", "seasonality", "generic"]
    nb_blocks_per_stack: int = 3
    thetas_dim: list[int] = [4, 8, 8]
    hidden_layer_units: int = 256
    share_weights_in_stack: bool = False
    nb_harmonics: int | None = None


def build_model(cfg, params):
    """Construct NBeats from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], stack_types=tuple(params.get('stack_types', ['trend', 'seasonality', 'generic'])), nb_blocks_per_stack=params.get('nb_blocks_per_stack', 3), thetas_dim=tuple(params.get('thetas_dim', [4, 8, 8])), hidden_layer_units=params.get('hidden_layer_units', 256), share_weights_in_stack=bool(params.get('share_weights_in_stack', False)), nb_harmonics=params.get('nb_harmonics', None))
    )


SPEC = ModelSpec(
    name='NBeats',
    module='models.nbeats',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='N-BEATS: Neural basis expansion analysis for interpretable time series forecasting',
        venue='ICLR 2020',
        year=2020,
        url='https://arxiv.org/abs/1905.10437',
    ),
    source=SourceRef(
        url='https://github.com/philipperemy/n-beats',
        revision='06a4e209ada80bf1f403ced5228261784dfb26ed',
        license='MIT',
    ),
    evidence="upstream-port",
    config_path='configs/models/NBeats.toml',
    model_card='src/models/nbeats/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The implementation is ported from the pinned third-party PyTorch reference, not the paper authors ServiceNow repository.',
        'Compile, fit, predict, checkpoint, and generator helpers are omitted; the architecture is exposed through the ModernTSF tensor contract.',
        'Channels are folded into the batch dimension and share one univariate stack; MS output retains only the final target channel.',
        'The independent backcast projection in the final generic block is frozen because the terminal residual is not consumed by the forecast objective.',
        'Published M3/M4/Tourism training recipes and numerical parity are not reproduced by the default config.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
