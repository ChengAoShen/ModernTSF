---
model: "XGBoostTS"
forecasting_setting: "time_series"
config: "configs/models/XGBoostTS.toml"
registry: "models.xgboost_ts.registry"
paper_title: "XGBoost: A Scalable Tree Boosting System"
venue: "KDD 2016"
year: 2016
arxiv: "https://arxiv.org/abs/1603.02754"
---
# XGBoostTS

XGBoostTS is a PyTorch-native adapter that implements an XGBoost-style gradient-boosted soft-tree ensemble for time series forecasting. It wraps the residual soft-tree boosting approach as a torch.nn.Module, enabling GPU/MPS execution through the standard ModernTSF trainer. The model operates on flattened lag features from the lookback window and produces direct multi-step forecasts.

## Paper
- **Title**: XGBoost: A Scalable Tree Boosting System
- **Venue**: KDD 2016
- **Published**: 2016 (arXiv: 2016-03)
- **arXiv**: https://arxiv.org/abs/1603.02754

## Abstract
Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

## In ModernTSF
Default config: `configs/models/XGBoostTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
