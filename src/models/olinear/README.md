---
model: "OLinear"
forecasting_setting: "time_series"
config: "configs/models/OLinear.toml"
registry: "models.olinear.registry"
paper_title: "OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2505.08550"
---
# OLinear

OLinear is a linear-based multivariate time series forecasting model that operates in an orthogonally transformed domain rather than directly in the time domain. It introduces OrthoTrans, a data-adaptive transformation built on an orthogonal matrix that diagonalizes the series' temporal Pearson correlation matrix via eigenvalue decomposition, yielding a decorrelated feature space for linear encoding. Complementing this, OLinear uses NormLin, a customized linear layer with a normalized weight matrix to capture multivariate dependencies, which empirically outperforms multi-head self-attention while requiring roughly half the FLOPs.

## Paper
- **Title**: OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.08550

## Abstract
This paper presents OLinear, a linear-based multivariate time series forecasting model that operates in an orthogonally transformed domain. Recent forecasting models typically adopt the temporal forecast (TF) paradigm, which directly encode and decode time series in the time domain. However, the entangled step-wise dependencies in series data can hinder the performance of TF. To address this, some forecasters conduct encoding and decoding in the transformed domain using fixed, dataset-independent bases (e.g., sine and cosine signals in the Fourier transform). In contrast, we utilize OrthoTrans, a data-adaptive transformation based on an orthogonal matrix that diagonalizes the series' temporal Pearson correlation matrix. This approach enables more effective encoding and decoding in the decorrelated feature domain and can serve as a plug-in module to enhance existing forecasters. To enhance the representation learning for multivariate time series, we introduce a customized linear layer, NormLin, which employs a normalized weight matrix to capture multivariate dependencies. Empirically, the NormLin module shows a surprising performance advantage over multi-head self-attention, while requiring nearly half the FLOPs. Extensive experiments on 24 benchmarks and 140 forecasting tasks demonstrate that OLinear consistently achieves state-of-the-art performance with high efficiency. Notably, as a plug-in replacement for self-attention, the NormLin module consistently enhances Transformer-based forecasters.

## In ModernTSF
Default config: `configs/models/OLinear.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2505-08550,
  author       = {Wenzhen Yue and
                  Yong Liu and
                  Haoxuan Li and
                  Hao Wang and
                  Xianghua Ying and
                  Ruohao Guo and
                  Bowei Xing and
                  Ji Shi},
  title        = {OLinear: {A} Linear Model for Time Series Forecasting in Orthogonally
                  Transformed Domain},
  journal      = {CoRR},
  volume       = {abs/2505.08550},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2505.08550},
  doi          = {10.48550/ARXIV.2505.08550},
  eprinttype   = {arXiv},
  eprint       = {2505.08550},
  timestamp    = {Fri, 08 May 2026 07:40:48 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2505-08550.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
