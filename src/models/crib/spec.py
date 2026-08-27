"""Model specification for CRIB."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.crib.model import Model

from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Parameters for the CRIB model.

    Note: ``patch_len`` must divide ``task.seq_len``; ``model_dim`` must be
    divisible by ``heads_num``.
    """

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=8, gt=0)
    model_dim: int = Field(default=32, gt=0)
    heads_num: int = Field(default=4, gt=0)
    enc_num: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["relu", "gelu"] = "relu"
    # Consistency (MSE) and IB (KL) regularizer weights (upstream 1.0 / 1e-6).
    consis_weight: float = Field(default=1.0, ge=0.0)
    kl_weight: float = Field(default=1e-6, ge=0.0)


def build_model(cfg, params):
    """Construct CRIB from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], patch_len=params.get('patch_len', 8), model_dim=params.get('model_dim', 32), heads_num=params.get('heads_num', 4), enc_num=params.get('enc_num', 3), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'relu'), consis_weight=params.get('consis_weight', 1.0), kl_weight=params.get('kl_weight', 1e-06))
    )


SPEC = ModelSpec(
    name='CRIB',
    module='models.crib',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Revisiting Multivariate Time Series Forecasting with Missing Values',
        venue='ICLR 2026',
        year=2026,
        url='https://arxiv.org/abs/2509.23494',
    ),
    source=SourceRef(url='https://github.com/Muyiiiii/CRIB', revision='a457672c7b0152f74c929858dba2a9c886405519', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/CRIB.toml',
    model_card='src/models/crib/README.md',
    smoke_config='configs/runs/smoke_crib.toml',
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'The patch encoder, TCN plus unified-variate attention, information-bottleneck latent, prediction head, consistency term, and KL term were compared with the pinned author repository.',
        'ModernTSF exposes the consistency and KL regularizers through aux_loss and expects the configured primary loss to supply the prediction term.',
        'The author missing-value masking and augmentation data pipeline is not included; the adapter operates on ordinary complete forecast windows, equivalent to missing_rate=0.',
        'The author repository has no explicit code license and no parity checkpoint; verification remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
