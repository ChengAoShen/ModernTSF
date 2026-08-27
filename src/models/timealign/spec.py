"""Model specification for TimeAlign."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timealign.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Parameters for the faithful TimeAlign model.

    Note: ``patch_num`` must divide both ``task.seq_len`` and ``task.pred_len``.
    """

    enc_in: int
    patch_num: int = 4
    d_model: int = 32
    d_ff: int = 32
    e_layers: int = 2
    dropout: float = 0.1
    pos: bool = True
    layer_norm: bool = True
    loc: bool = True
    glo: bool = True
    local_margin: float = 0.0
    global_margin: float = 0.0
    w_recon: float = 1.0
    w_align: float = 0.1


def build_model(cfg, params):
    """Construct TimeAlign from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], patch_num=params.get('patch_num', 4), d_model=params.get('d_model', 32), d_ff=params.get('d_ff', 32), e_layers=params.get('e_layers', 2), dropout=params.get('dropout', 0.1), pos=bool(params.get('pos', True)), layer_norm=bool(params.get('layer_norm', True)), loc=bool(params.get('loc', True)), glo=bool(params.get('glo', True)), local_margin=params.get('local_margin', 0.0), global_margin=params.get('global_margin', 0.0), w_recon=params.get('w_recon', 1.0), w_align=params.get('w_align', 0.1))
    )


SPEC = ModelSpec(
    name='TimeAlign',
    module='models.timealign',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting',
        venue='ICLR 2026',
        year=2026,
        url='https://arxiv.org/abs/2509.14181',
    ),
    source=SourceRef(
        url='https://github.com/TROUBADOUR000/TimeAlign',
        revision='ab2dff5bde250f82e29d8755f87a494921857d71',
        license='NOASSERTION',
    ),
    evidence="upstream-port",
    config_path='configs/models/TimeAlign.toml',
    model_card='src/models/timealign/README.md',
    smoke_config='configs/runs/smoke_timealign.toml',
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The author repository declares no license; the pinned revision is recorded for inspection, not as a redistribution grant.',
        'Patch embedding, history/future encoders, local/global alignment, reconstruction projection, and normalization follow the pinned author implementation.',
        'ModernTSF supplies the future target through the trainer hook and exposes the full prediction, reconstruction, and alignment objective through train_loss_override.',
        'The upstream runner, datasets, shell presets, and published numerical results are not reproduced.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
