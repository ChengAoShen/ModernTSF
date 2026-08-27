"""Model specification for GlocalIB."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.glocalib.model import Model

from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Parameters for the Glocal-IB forecasting model."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    # Alignment regularizer weight and augmentation strength.
    align_weight: float = Field(default=0.5, ge=0.0)
    mask_ratio: float = Field(default=0.25, ge=0.0, lt=1.0)
    # "cos_align" (robust) or "contrastive" (InfoNCE over time steps).
    align_loss_type: Literal["cos_align", "contrastive"] = "cos_align"


def build_model(cfg, params):
    """Construct GlocalIB from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), align_weight=params.get('align_weight', 0.5), mask_ratio=params.get('mask_ratio', 0.25), align_loss_type=params.get('align_loss_type', 'cos_align'))
    )


SPEC = ModelSpec(
    name='GlocalIB',
    module='models.glocalib',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Glocal Information Bottleneck for Time Series Imputation',
        venue='NeurIPS 2025',
        year=2025,
        url='https://arxiv.org/abs/2510.04910',
    ),
    source=SourceRef(
        url='https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB',
        revision='1ee232e6d6b28329010db0305899511cb7fc9016',
        license='NOASSERTION',
    ),
    evidence="adaptation",
    config_path='configs/models/GlocalIB.toml',
    model_card='src/models/glocalib/README.md',
    smoke_config='configs/runs/smoke_glocalib.toml',
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'The source repository declares no license; its revision is recorded for inspection, not as a redistribution grant.',
        'The paper and upstream code target imputation, while this local model adapts the projector/stop-gradient alignment mechanism to forecasting.',
        'A lightweight local forecaster replaces the upstream imputation backbones; random temporal masking provides the corrupted view.',
        'The alignment term is exposed through ModernTSF aux_loss; paper numerical parity is not claimed.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
