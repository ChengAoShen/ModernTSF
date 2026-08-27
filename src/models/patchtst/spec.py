"""Model specification for PatchTST."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.patchtst.model import Model

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    e_layers: int = 3
    d_model: int = 512
    n_heads: int = 8
    d_k: Optional[int] = None
    d_v: Optional[int] = None
    d_ff: int = 2048
    activation: str = "gelu"
    norm: str = "BatchNorm"
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    res_dropout: float = 0.0
    proj_dropout: float = 0.0
    head_dropout: float = 0.0
    pre_norm: bool = False
    pe: str = "zeros"
    learn_pe: bool = False
    individual: bool = False
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False


def build_model(cfg, params):
    """Construct PatchTST from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], context_window=cfg.task.seq_len, target_window=cfg.task.pred_len, patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), padding_patch=params.get('padding_patch', 'end'), n_layers=params.get('e_layers', 3), d_model=params.get('d_model', 512), n_heads=params.get('n_heads', 8), d_k=params.get('d_k'), d_v=params.get('d_v'), d_ff=params.get('d_ff', 2048), activation=params.get('activation', 'gelu'), norm=params.get('norm', 'BatchNorm'), attn_dropout=params.get('attn_dropout', 0.0), res_dropout=params.get('res_dropout', 0.0), ffn_dropout=params.get('ffn_dropout', 0.0), proj_dropout=params.get('proj_dropout', 0.0), head_dropout=params.get('head_dropout', 0.0), pre_norm=bool(params.get('pre_norm', False)), pe=params.get('pe', 'zeros'), learn_pe=bool(params.get('learn_pe', False)), head_type='flatten', individual=bool(params.get('individual', False)), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', False)), subtract_last=bool(params.get('subtract_last', False)))
    )


SPEC = ModelSpec(
    name='PatchTST',
    module='models.patchtst',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='A Time Series is Worth 64 Words: Long-term Forecasting with Transformers',
        venue='ICLR 2023',
        year=2023,
        url='https://openreview.net/forum?id=Jbdc0vTOcol',
    ),
    source=SourceRef(
        url='https://github.com/yuqinie98/PatchTST',
        revision='204c21efe0b39603ad6e2ca640ef5896646ab1a9',
        license='Apache-2.0',
    ),
    evidence="adaptation",
    config_path='configs/models/PatchTST.toml',
    model_card='src/models/patchtst/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('patchtst',),
    deviations=(
        'The shared backbone preserves supervised PatchTST patching, channel independence, shared projection and Transformer weights, RevIN, and flatten forecasting head from the pinned author repository.',
        'The component was reorganized across patchtst, tst_transformer, positional_encoding, and revin modules and compared with THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e.',
        'The local encoder omits the optional residual-attention score path enabled by default in the author supervised backbone and exposes separate residual, feed-forward, attention, and projection dropout rates.',
        'Only the flatten forecasting head is implemented; the previously exposed head_type string was inert and has been removed.',
        'The self-supervised pretraining, masking, transfer-learning paths, dataset preprocessing, loss, optimizer, and published checkpoints are not included.',
        'The runnable preset uses patch length 16 and stride 8 with d_model=512, n_heads=8, d_ff=2048, learn_pe=false, and affine=false; these differ from the author backbone defaults and do not claim parity with every paper configuration.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
