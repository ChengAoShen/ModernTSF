---
model: "Pathformer"
forecasting_setting: "time_series"
config: "configs/models/Pathformer.toml"
registry: "models.pathformer.registry"
paper_title: "Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2402.05956"
---
# Pathformer

Pathformer is a multi-scale Transformer for multivariate time-series forecasting that integrates temporal resolution and temporal distance in a unified framework. It divides the input series into patches of multiple sizes (multi-scale division), applies dual attention over each scale to capture both global correlations and local details, and routes the information through adaptive pathways that dynamically adjust the multi-scale modelling process based on the varying temporal dynamics of each input.

## Paper
- **Title**: Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.05956

## Abstract
Transformers for time series forecasting mainly model time series from limited or fixed scales, making it challenging to capture different characteristics spanning various scales. We propose Pathformer, a multi-scale Transformer with adaptive pathways. It integrates both temporal resolution and temporal distance for multi-scale modeling. Multi-scale division divides the time series into different temporal resolutions using patches of various sizes. Based on the division of each scale, dual attention is performed over these patches to capture global correlations and local details as temporal dependencies. We further enrich the multi-scale Transformer with adaptive pathways, which adaptively adjust the multi-scale modeling process based on the varying temporal dynamics of the input, improving the accuracy and generalization of Pathformer. Extensive experiments on eleven real-world datasets demonstrate that Pathformer not only achieves state-of-the-art performance by surpassing all current models but also exhibits stronger generalization abilities under various transfer scenarios.

## In ModernTSF
Default config: `configs/models/Pathformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/ChenZ0SWW0G24,
  author       = {Peng Chen and
                  Yingying Zhang and
                  Yunyao Cheng and
                  Yang Shu and
                  Yihang Wang and
                  Qingsong Wen and
                  Bin Yang and
                  Chenjuan Guo},
  title        = {Pathformer: Multi-scale Transformers with Adaptive Pathways for Time
                  Series Forecasting},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=lJkOCMP2aW},
  timestamp    = {Tue, 12 Aug 2025 11:51:29 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/ChenZ0SWW0G24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
