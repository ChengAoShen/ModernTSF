---
model: "CARD"
forecasting_setting: "time_series"
config: "configs/models/CARD.toml"
registry: "models.card.registry"
paper_title: "CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2305.12095"
---
# CARD

CARD (Channel Aligned Robust Blend Transformer) is a Transformer-based model for multivariate long-term and short-term time series forecasting. It addresses the limitations of channel-independent Transformers by introducing a channel-aligned attention structure that jointly captures temporal correlations and cross-variable dependencies, a token blend module for multi-scale feature extraction, and a robust uncertainty-weighted loss function to reduce overfitting.

## Paper
- **Title**: CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.12095

## Abstract
Recent studies have demonstrated the great power of Transformer models for time series forecasting. One of the key elements that lead to the transformer's success is the channel-independent (CI) strategy to improve the training robustness. However, the ignorance of the correlation among different channels in CI would limit the model's forecasting capacity. In this work, we design a special Transformer, i.e., Channel Aligned Robust Blend Transformer (CARD for short), that addresses key shortcomings of CI type Transformer in time series forecasting. First, CARD introduces a channel-aligned attention structure that allows it to capture both temporal correlations among signals and dynamical dependence among multiple variables over time. Second, in order to efficiently utilize the multi-scale knowledge, we design a token blend module to generate tokens with different resolutions. Third, we introduce a robust loss function for time series forecasting to alleviate the potential overfitting issue. This new loss function weights the importance of forecasting over a finite horizon based on prediction uncertainties. Our evaluation of multiple long-term and short-term forecasting datasets demonstrates that CARD significantly outperforms state-of-the-art time series forecasting methods. The code is available at the following repository: https://github.com/wxie9/CARD

## In ModernTSF
Default config: `configs/models/CARD.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/WangZWGD024,
  author       = {Xue Wang and
                  Tian Zhou and
                  Qingsong Wen and
                  Jinyang Gao and
                  Bolin Ding and
                  Rong Jin},
  title        = {{CARD:} Channel Aligned Robust Blend Transformer for Time Series Forecasting},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=MJksrOhurE},
  timestamp    = {Thu, 23 Jan 2025 19:51:39 +0100},
  biburl       = {https://dblp.org/rec/conf/iclr/WangZWGD024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
