---
model: "PMDformer"
forecasting_setting: "time_series"
config: "configs/models/PMDformer.toml"
registry: "models.pmdformer.registry"
paper_title: "PMDformer: Patch-Mean Decoupling Transformer for Long-term Forecasting"
venue: "ICLR 2026"
year: 2026
arxiv: ""
---
# PMDformer

PMDformer is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It decouples patch-level local shape fluctuations from their mean (trend) level through Patch-Mean Decoupling (PMD), combines Proximal Variable Attention (PVA) to focus on the most relevant inter-variable interactions, and applies Trend Recovery Attention (TRA) to restore long-term trend information, improving both forecasting accuracy and computational efficiency.

## Paper
- **Title**: PMDformer: Patch-Mean Decoupling Transformer for Long-term Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
The official paper abstract is not available on arXiv. According to publicly available information about the ICLR 2026 accepted paper (https://github.com/aohu1105/PMDformer), PMDformer introduces three core innovations: (1) Patch-Mean Decoupling (PMD), which separates local shape fluctuations from their absolute magnitude (mean level) to reduce bias and better capture underlying patterns; (2) Proximal Variable Attention (PVA), which strengthens focus on the most relevant and temporally proximal inter-variable interactions; and (3) Trend Recovery Attention (TRA), which restores long-term trend information to improve both responsiveness and stability in forecasting. Together, these components deliver stronger forecasting accuracy and stability while reducing memory usage compared to previous patch-based Transformer methods.

## In ModernTSF
Default config: `configs/models/PMDformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{hu2026pmdformer,
  author    = {Ao Hu and Liangjian Wen and Jiang Duan and Yong Dai and Yan He and Dongkai Wang and Jun Wang and Yukun Zhang and Ruoxi Jiang and Zenglin Xu},
  title     = {{PMD}former: Patch-Mean Decoupling Transformer for Long-term Forecasting},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rfJ41gK9Ct}
}
```
