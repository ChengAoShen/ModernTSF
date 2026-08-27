---
name: "iTransformer"
implementation: rewrite
summary: "iTransformer is a Transformer-based model for multivariate time series forecasting that inverts the conventional token design: instead of embedding multiple variates at the same timestamp into one token, it embeds the entire time series of each individual variate into a single variate token. Attention is then applied across variates to capture inter-channel correlations, while the feed-forward network learns nonlinear temporal representations per variate."
paper:
  title: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
  venue: "ICLR 2024"
  year: 2024
  url: "https://arxiv.org/abs/2310.06625"
codebase:
  url: "https://github.com/thuml/iTransformer"
  revision: "c2426e68ca13f74aaec08045c5c724d8ad328124"
  license: "MIT"
  usage: reference-only
---
# iTransformer

iTransformer is a Transformer-based model for multivariate time series forecasting that inverts the conventional token design: instead of embedding multiple variates at the same timestamp into one token, it embeds the entire time series of each individual variate into a single variate token. Attention is then applied across variates to capture inter-channel correlations, while the feed-forward network learns nonlinear temporal representations per variate.

<!-- model-card:canonical:start -->
## Method overview

iTransformer is a Transformer-based model for multivariate time series forecasting that inverts the conventional token design: instead of embedding multiple variates at the same timestamp into one token, it embeds the entire time series of each individual variate into a single variate token.

## Core architecture

Attention is then applied across variates to capture inter-channel correlations, while the feed-forward network learns nonlinear temporal representations per variate.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2310.06625); title: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/thuml/iTransformer); revision: `c2426e68ca13f74aaec08045c5c724d8ad328124`; license: `MIT`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/iTransformer.toml`](../../../configs/models/iTransformer.toml).

## Differences

Compared against the author repository at commit `c2426e68ca13f74aaec08045c5c724d8ad328124` (MIT). Inverted variate tokens, attention across variates, normalization, and the projection head are retained; non-forecast tasks are omitted. The inert `class_strategy` option was removed.

## Shared components

- [`embed`](../../components/embed.py)
- [`self_attention_family`](../../components/self_attention_family.py)
- [`transformer_encdec`](../../components/transformer_encdec.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `freq='h'`, `embed='timeF'`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_ff=2048`, `factor=1`, `dropout=0.1`, `activation='gelu'`, `output_attention=False`, `use_norm=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- **Venue**: ICLR 2024
- **Published**: 2024 (arXiv: 2023-10)
- **arXiv**: https://arxiv.org/abs/2310.06625

## Abstract
The recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters. These forecasters leverage Transformers to model the global dependencies over temporal tokens of time series, with each token formed by multiple variates of the same timestamp. However, Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. Besides, the embedding for each temporal token fuses multiple variates that represent potential delayed events and distinct physical measurements, which may fail in learning variate-centric representations and result in meaningless attention maps. In this work, we reflect on the competent duties of Transformer components and repurpose the Transformer architecture without any modification to the basic components. We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions. Specifically, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations. The iTransformer model achieves state-of-the-art on challenging real-world datasets, which further empowers the Transformer family with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows, making it a nice alternative as the fundamental backbone of time series forecasting.

## In ModernTSF
Default config: `configs/models/iTransformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

Compared against the author repository at commit `c2426e68ca13f74aaec08045c5c724d8ad328124` (MIT). Inverted variate tokens, attention across variates, normalization, and the projection head are retained; non-forecast tasks are omitted. The inert `class_strategy` option was removed.

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
