"""Model specification for ModernTCN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.moderntcn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    ffn_ratio: int = 1
    num_blocks: list[int] = [1]
    large_size: list[int] = [13]
    small_size: list[int] = [5]
    dims: list[int] = [32]
    dw_dims: list[int] = [32]
    patch_size: int = 16
    patch_stride: int = 16
    stem_ratio: int = 6
    downsample_ratio: int = 2
    small_kernel_merged: bool = False
    dropout: float = 0.1
    head_dropout: float = 0.1
    use_multi_scale: bool = True
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    decomposition: bool = False
    kernel_size: int = 25


def build_model(cfg, params):
    """Construct ModernTCN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], ffn_ratio=params.get('ffn_ratio', 1), num_blocks=params.get('num_blocks', [1]), large_size=params.get('large_size', [13]), small_size=params.get('small_size', [5]), dims=params.get('dims', [32]), dw_dims=params.get('dw_dims', [32]), patch_size=params.get('patch_size', 16), patch_stride=params.get('patch_stride', 16), stem_ratio=params.get('stem_ratio', 6), downsample_ratio=params.get('downsample_ratio', 2), small_kernel_merged=bool(params.get('small_kernel_merged', False)), dropout=params.get('dropout', 0.1), head_dropout=params.get('head_dropout', 0.1), use_multi_scale=bool(params.get('use_multi_scale', True)), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', True)), subtract_last=bool(params.get('subtract_last', False)), individual=bool(params.get('individual', False)), decomposition=bool(params.get('decomposition', False)), kernel_size=params.get('kernel_size', 25))
    )


SPEC = ModelSpec(
    name='ModernTCN',
    module='models.moderntcn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis',
        venue='ICLR 2024',
        year=2024,
        url='https://openreview.net/forum?id=vpJMJerXHU',
    ),
    source=SourceRef(url='https://github.com/luodhhh/ModernTCN', revision='56a9a2c018385cd5acef015378cae7f084d1b11c', license='MIT'),
    evidence="adaptation",
    config_path='configs/models/ModernTCN.toml',
    model_card='src/models/moderntcn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('flatten_forecast_head', 'revin', 'series_decomposition'),
    deviations=(
        'The long-term forecast core was adapted from the pinned author repository and compared with THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e.',
        'Patch stem, multi-stage downsampling, reparameterizable large/small depthwise kernels, variable-independent and variable-mixing FFNs, multi-scale head, RevIN, and optional decomposition are retained.',
        'The upstream time-feature embedding branch and non-forecast tasks are removed; temporal marks are not consumed by this adapter.',
        'patch_size must divide seq_len to avoid source-style silent truncation; architecture list lengths must agree.',
        'Published training protocol and numerical parity are not reproduced.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
