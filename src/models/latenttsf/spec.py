"""Model specification for LatentTSF."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.latenttsf.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Parameters for the faithful two-stage LatentTSF model."""

    enc_in: int
    # Latent state dimension and AE hidden width.
    d_model: int = 64
    d_ff: int = 128
    # Latent objective weights (paper Sec. 5.3.2): alpha (Pred), beta (Align).
    mse_weight: float = 10.0
    cosine_weight: float = 15.0
    use_latent_norm: bool = True
    # DLinear backbone (paper's primary Table-1 backbone) hyper-parameters.
    kernel_size: int = 25
    individual: bool = False
    # Stage-1 AE pretraining schedule (used when autoencoder_path is empty).
    # ae_loss is "MAE" (L1) or "MSE".
    ae_train_epochs: int = 100
    ae_lr: float = 5e-4
    ae_loss: str = "MAE"
    # Optional pretrained MLP-AE checkpoint.pth (or its folder). When set,
    # Stage 1 loads + freezes it instead of pretraining on the fly. d_model /
    # d_ff / enc_in must match the checkpoint.
    autoencoder_path: str = ""


def build_model(cfg, params):
    """Construct LatentTSF from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), d_ff=params.get('d_ff', 128), mse_weight=params.get('mse_weight', 10.0), cosine_weight=params.get('cosine_weight', 15.0), use_latent_norm=bool(params.get('use_latent_norm', True)), kernel_size=params.get('kernel_size', 25), individual=bool(params.get('individual', False)), ae_train_epochs=params.get('ae_train_epochs', 100), ae_lr=params.get('ae_lr', 0.0005), ae_loss=params.get('ae_loss', 'MAE'), autoencoder_path=params.get('autoencoder_path', ''))
    )


SPEC = ModelSpec(
    name='LatentTSF',
    module='models.latenttsf',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='From Observations to States: Latent Time Series Forecasting',
        venue='ICML 2026',
        year=2026,
        url='https://arxiv.org/abs/2602.00297',
    ),
    source=SourceRef(
        url='https://github.com/Muyiiiii/LatentTSF',
        revision='7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2',
        license='MIT',
    ),
    evidence="adaptation",
    config_path='configs/models/LatentTSF.toml',
    model_card='src/models/latenttsf/README.md',
    smoke_config='configs/runs/smoke_latenttsf.toml',
    capabilities=frozenset(['time-series']),
    components=('dlinear',),
    deviations=(
        'The two-stage frozen-autoencoder and latent forecasting objective follow the pinned author repository, but trainer integration is a ModernTSF adaptation rather than a file-level port.',
        'The local implementation supports the paper primary DLinear latent forecaster only and reuses the shared DLinear component.',
        'Autoencoder pretraining defaults to 100 epochs instead of the upstream 500; a pinned pretrained checkpoint may be supplied explicitly.',
        'Published dataset-specific latent widths, learning rates, checkpoints, and numerical results are not reproduced by the default config.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
