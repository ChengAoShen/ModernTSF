---
model: "NSTransformer"
forecasting_setting: "time_series"
config: "configs/models/NSTransformer.toml"
registry: "models.nstransformer.registry"
paper_title: "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting"
venue: "NeurIPS 2022"
year: 2022
arxiv: "https://arxiv.org/abs/2205.14415"
---
# NSTransformer

NSTransformer (Non-stationary Transformer) is a time series forecasting model that addresses the over-stationarization problem in Transformer-based forecasters. It augments any standard Transformer backbone with two interdependent modules — Series Stationarization, which normalises input statistics and restores them in the output for improved predictability, and De-stationary Attention, which recovers intrinsic non-stationary information into the computed temporal dependencies by approximating distinguishable attentions learned from the raw, un-normalised series.

## Paper
- **Title**: Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
- **Venue**: NeurIPS 2022
- **Published**: 2022 (arXiv: 2022-05)
- **arXiv**: https://arxiv.org/abs/2205.14415

## Abstract
Transformers have shown great power in time series forecasting due to their global-range modeling ability. However, their performance can degenerate terribly on non-stationary real-world data in which the joint distribution changes over time. Previous studies primarily adopt stationarization to attenuate the non-stationarity of original series for better predictability. But the stationarized series deprived of inherent non-stationarity can be less instructive for real-world bursty events forecasting. This problem, termed over-stationarization in this paper, leads Transformers to generate indistinguishable temporal attentions for different series and impedes the predictive capability of deep models. To tackle the dilemma between series predictability and model capability, we propose Non-stationary Transformers as a generic framework with two interdependent modules: Series Stationarization and De-stationary Attention. Concretely, Series Stationarization unifies the statistics of each input and converts the output with restored statistics for better predictability. To address the over-stationarization problem, De-stationary Attention is devised to recover the intrinsic non-stationary information into temporal dependencies by approximating distinguishable attentions learned from raw series. Our Non-stationary Transformers framework consistently boosts mainstream Transformers by a large margin, which reduces MSE by 49.43% on Transformer, 47.34% on Informer, and 46.89% on Reformer, making them the state-of-the-art in time series forecasting.

## In ModernTSF
Default config: `configs/models/NSTransformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/LiuWWL22,
  author       = {Yong Liu and
                  Haixu Wu and
                  Jianmin Wang and
                  Mingsheng Long},
  editor       = {Sanmi Koyejo and
                  S. Mohamed and
                  A. Agarwal and
                  Danielle Belgrave and
                  K. Cho and
                  A. Oh},
  title        = {Non-stationary Transformers: Exploring the Stationarity in Time Series
                  Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 35: Annual Conference
                  on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans,
                  LA, USA, November 28 - December 9, 2022},
  year         = {2022},
  url          = {http://papers.nips.cc/paper\_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html},
  timestamp    = {Sun, 29 Mar 2026 11:26:46 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/LiuWWL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
