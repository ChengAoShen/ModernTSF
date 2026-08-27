"""Model specification for HL."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.hl.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """HL has no tunable parameters beyond enc_in (num nodes)."""

    enc_in: int = 207


def build_model(cfg, params):
    """Construct HL from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'])
    )


SPEC = ModelSpec(
    name='HL',
    module='models.hl',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Historical-last persistence baseline (no associated paper)',
        venue='N/A (classical baseline)',
        year=None,
        url='',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/CauAir',
        revision='73dae00ca6ad14abb15174a0a0286d500e868b94',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/HL.toml',
    model_card='src/models/hl/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'No canonical paper or author-owned reference implementation was identified.',
        'The immediate CauAir source repository does not declare a license.',
        'A dummy linear path contributes exactly zero and exists only for the shared optimizer/backward contract.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
