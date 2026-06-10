---
model: "iTransformer"
forecasting_setting: "time_series"
config: "configs/models/iTransformer.toml"
registry: "models.itransformer.registry"
paper_title: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2310.06625"
---
# iTransformer

iTransformer is a Transformer-based model for multivariate time series forecasting that inverts the conventional token design: instead of embedding multiple variates at the same timestamp into one token, it embeds the entire time series of each individual variate into a single variate token. Attention is then applied across variates to capture inter-channel correlations, while the feed-forward network learns nonlinear temporal representations per variate.

## Paper
- **Title**: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2023-10)
- **arXiv**: https://arxiv.org/abs/2310.06625

## Abstract
The recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters. These forecasters leverage Transformers to model the global dependencies over temporal tokens of time series, with each token formed by multiple variates of the same timestamp. However, Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. Besides, the embedding for each temporal token fuses multiple variates that represent potential delayed events and distinct physical measurements, which may fail in learning variate-centric representations and result in meaningless attention maps. In this work, we reflect on the competent duties of Transformer components and repurpose the Transformer architecture without any modification to the basic components. We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions. Specifically, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations. The iTransformer model achieves state-of-the-art on challenging real-world datasets, which further empowers the Transformer family with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows, making it a nice alternative as the fundamental backbone of time series forecasting.

## In ModernTSF
Default config: `configs/models/iTransformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LiuHZWWML24,
  author       = {Yong Liu and
                  Tengge Hu and
                  Haoran Zhang and
                  Haixu Wu and
                  Shiyu Wang and
                  Lintao Ma and
                  Mingsheng Long},
  title        = {iTransformer: Inverted Transformers Are Effective for Time Series
                  Forecasting},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=JePfAI8fah},
  timestamp    = {Sun, 29 Mar 2026 11:26:46 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LiuHZWWML24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
