---
name: "PatchTST"
implementation: rewrite
summary: "PatchTST is a Transformer-based model for multivariate and univariate long-term time-series forecasting that segments each channel into subseries-level patches fed as input tokens, combined with a channel-independence strategy where each channel shares the same Transformer weights. This design retains local semantic information, drastically reduces attention-map memory, and allows the model to attend over a much longer historical context."
paper:
  title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
  venue: "ICLR 2023"
  year: 2023
  url: "https://openreview.net/forum?id=Jbdc0vTOcol"
codebase:
  url: "https://github.com/yuqinie98/PatchTST"
  revision: "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
  license: "Apache-2.0"
  usage: reference-only
---
# PatchTST

PatchTST is a Transformer-based model for multivariate and univariate long-term time-series forecasting that segments each channel into subseries-level patches fed as input tokens, combined with a channel-independence strategy where each channel shares the same Transformer weights. This design retains local semantic information, drastically reduces attention-map memory, and allows the model to attend over a much longer historical context.

<!-- model-card:canonical:start -->
## Method overview

PatchTST is a Transformer-based model for multivariate and univariate long-term time-series forecasting that segments each channel into subseries-level patches fed as input tokens, combined with a channel-independence strategy where each channel shares the same Transformer weights.

## Core architecture

This design retains local semantic information, drastically reduces attention-map memory, and allows the model to attend over a much longer historical context.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=Jbdc0vTOcol); title: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers; venue/year: ICLR 2023 / 2023
- [codebase](https://github.com/yuqinie98/PatchTST); revision: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`; license: `Apache-2.0`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/PatchTST.toml`](../../../configs/models/PatchTST.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

- [`patchtst`](../../components/patchtst.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=16`, `stride=8`, `padding_patch='end'`, `e_layers=3`, `d_model=512`, `n_heads=8`, `d_ff=2048`, `activation='gelu'`, `norm='BatchNorm'`, `attn_dropout=0.0`, `ffn_dropout=0.0`, `res_dropout=0.0`, `proj_dropout=0.0`, `head_dropout=0.0`, `pre_norm=False`, `pe='zeros'`, `learn_pe=False`, `individual=False`, `revin=True`, `affine=False`, `subtract_last=False`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Venue**: ICLR 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.14730

## Abstract
We propose an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is based on two key components: (i) segmentation of time series into subseries-level patches which are served as input tokens to Transformer; (ii) channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series. Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history. Our channel-independent patch time series Transformer (PatchTST) can improve the long-term forecasting accuracy significantly when compared with that of SOTA Transformer-based models. We also apply our model to self-supervised pre-training tasks and attain excellent fine-tuning performance, which outperforms supervised training on large datasets. Transferring of masked pre-trained representation on one dataset to others also produces SOTA forecasting accuracy.

## In ModernTSF
Default config: `configs/models/PatchTST.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

This supervised forecasting entry is a **modified integration** of the Apache-2.0 author
repository at revision `204c21efe0b39603ad6e2ca640ef5896646ab1a9`, also
compared with THUML Time-Series-Library revision
`4e938a1767106324dd753b2a44832bf870a0252e`. Patching, channel independence,
shared encoder weights, RevIN, and the flatten head are retained. The local
encoder omits optional residual-attention score accumulation and provides
separate dropout controls. Only the flatten forecasting head is implemented;
self-supervised pretraining, transfer learning, and checkpoint parity are not.
The runnable width, head count, positional-encoding learning, and RevIN affine
defaults also differ from the author backbone defaults.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/NieNSK23,
  author       = {Yuqi Nie and
                  Nam H. Nguyen and
                  Phanwadee Sinthong and
                  Jayant Kalagnanam},
  title        = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/forum?id=Jbdc0vTOcol},
  timestamp    = {Wed, 24 Jul 2024 16:50:33 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/NieNSK23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
