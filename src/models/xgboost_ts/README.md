---
name: "XGBoostTS"
implementation: rewrite
summary: "XGBoostTS is a PyTorch-native adapter that implements an XGBoost-style gradient-boosted soft-tree ensemble for time series forecasting. It wraps the residual soft-tree boosting approach as a torch.nn.Module, enabling GPU/MPS execution through the standard ModernTSF trainer. The model operates on flattened lag features from the lookback window and produces direct multi-step forecasts."
paper:
  title: "XGBoost: A Scalable Tree Boosting System"
  venue: "KDD 2016"
  year: 2016
  url: "https://arxiv.org/abs/1603.02754"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# XGBoostTS

XGBoostTS is a PyTorch-native adapter that implements an XGBoost-style gradient-boosted soft-tree ensemble for time series forecasting. It wraps the residual soft-tree boosting approach as a torch.nn.Module, enabling GPU/MPS execution through the standard ModernTSF trainer. The model operates on flattened lag features from the lookback window and produces direct multi-step forecasts.

<!-- model-card:canonical:start -->
## Method overview

XGBoostTS is a PyTorch-native adapter that implements an XGBoost-style gradient-boosted soft-tree ensemble for time series forecasting.

## Core architecture

It wraps the residual soft-tree boosting approach as a torch.nn.Module, enabling GPU/MPS execution through the standard ModernTSF trainer. The model operates on flattened lag features from the lookback window and produces direct multi-step forecasts.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1603.02754); title: XGBoost: A Scalable Tree Boosting System; venue/year: KDD 2016 / 2016
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/XGBoostTS.toml`](../../../configs/models/XGBoostTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0001`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: XGBoost: A Scalable Tree Boosting System
- **Venue**: KDD 2016
- **Published**: 2016 (arXiv: 2016-03)
- **arXiv**: https://arxiv.org/abs/1603.02754

## Abstract
Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

## In ModernTSF
Default config: `configs/models/XGBoostTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/ChenG16,
  author       = {Tianqi Chen and
                  Carlos Guestrin},
  editor       = {Balaji Krishnapuram and
                  Mohak Shah and
                  Alexander J. Smola and
                  Charu C. Aggarwal and
                  Dou Shen and
                  Rajeev Rastogi},
  title        = {XGBoost: {A} Scalable Tree Boosting System},
  booktitle    = {Proceedings of the 22nd {ACM} {SIGKDD} International Conference on
                  Knowledge Discovery and Data Mining, San Francisco, CA, USA, August
                  13-17, 2016},
  pages        = {785--794},
  publisher    = {{ACM}},
  year         = {2016},
  url          = {https://doi.org/10.1145/2939672.2939785},
  doi          = {10.1145/2939672.2939785},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/ChenG16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
