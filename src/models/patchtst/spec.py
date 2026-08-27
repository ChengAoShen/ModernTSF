"""Model specification for PatchTST."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
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
    config_path='configs/models/PatchTST.toml',
    model_card='src/models/patchtst/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('patchtst',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
