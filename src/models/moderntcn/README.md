---
name: "ModernTCN"
implementation: rewrite
summary: "ModernTCN is a pure convolutional architecture for general time series analysis that modernizes the traditional Temporal Convolutional Network (TCN) by incorporating large effective receptive fields through depthwise separable convolutions, achieving state-of-the-art performance across long-term and short-term forecasting, imputation, classification, and anomaly detection tasks."
paper:
  title: "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"
  venue: "ICLR 2024"
  year: 2024
  url: "https://openreview.net/forum?id=vpJMJerXHU"
codebase:
  url: "https://github.com/luodhhh/ModernTCN"
  revision: "56a9a2c018385cd5acef015378cae7f084d1b11c"
  license: "MIT"
  usage: reference-only
---
# ModernTCN

ModernTCN is a pure convolutional architecture for general time series analysis that modernizes the traditional Temporal Convolutional Network (TCN) by incorporating large effective receptive fields through depthwise separable convolutions, achieving state-of-the-art performance across long-term and short-term forecasting, imputation, classification, and anomaly detection tasks.

<!-- model-card:canonical:start -->
## Method overview

ModernTCN is a pure convolutional architecture for general time series analysis that modernizes the traditional Temporal Convolutional Network (TCN) by incorporating large effective receptive fields through depthwise separable convolutions, achieving state-of-the-art performance across long-term and short-term forecasting, imputation, classification, and anomaly detection tasks.

## Core architecture

ModernTCN is a pure convolutional architecture for general time series analysis that modernizes the traditional Temporal Convolutional Network (TCN) by incorporating large effective receptive fields through depthwise separable convolutions, achieving state-of-the-art performance across long-term and short-term forecasting, imputation, classification, and anomaly detection tasks.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://openreview.net/forum?id=vpJMJerXHU); title: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis; venue/year: ICLR 2024 / 2024
- [codebase](https://github.com/luodhhh/ModernTCN); revision: `56a9a2c018385cd5acef015378cae7f084d1b11c`; license: `MIT`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ModernTCN.toml`](../../../configs/models/ModernTCN.toml).

## Differences

Compared against the author repository at commit `56a9a2c018385cd5acef015378cae7f084d1b11c` (MIT). Large-kernel depthwise blocks, patch stem, multi-scale stages, RevIN, and the forecast head are retained; time-feature embedding and non-forecast tasks from upstream are omitted.

## Shared components

- [`flatten_forecast_head`](../../components/flatten_forecast_head.py)
- [`revin`](../../components/revin.py)
- [`series_decomposition`](../../components/series_decomposition.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `ffn_ratio=1`, `num_blocks=[1]`, `large_size=[13]`, `small_size=[5]`, `dims=[64]`, `dw_dims=[64]`, `patch_size=16`, `patch_stride=16`, `stem_ratio=6`, `downsample_ratio=2`, `small_kernel_merged=False`, `dropout=0.1`, `head_dropout=0.1`, `use_multi_scale=True`, `revin=True`, `affine=True`, `subtract_last=False`, `individual=False`, `decomposition=False`, `kernel_size=25`
<!-- model-card:canonical:end -->

## Paper
- **Title**: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
- **Venue**: ICLR 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Recently, Transformer-based and MLP-based models have emerged rapidly and won dominance in time series analysis. In contrast, convolution is losing steam in time series tasks nowadays for inferior performance. This paper studies the open question of how to better use convolution in time series analysis and makes efforts to bring convolution back to the arena of time series analysis. To this end, we modernize the traditional TCN and conduct time series related modifications to make it more suitable for time series tasks. As the outcome, we propose ModernTCN and successfully solve this open question through a seldom-explored way in time series community. As a pure convolution structure, ModernTCN still achieves the consistent state-of-the-art performance on five mainstream time series analysis tasks while maintaining the efficiency advantage of convolution-based models, therefore providing a better balance of efficiency and performance than state-of-the-art Transformer-based and MLP-based models. Our study further reveals that, compared with previous convolution-based models, our ModernTCN has much larger effective receptive fields (ERFs), therefore can better unleash the potential of convolution in time series analysis.

## In ModernTSF
Default config: `configs/models/ModernTCN.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

Compared against the author repository at commit `56a9a2c018385cd5acef015378cae7f084d1b11c` (MIT). Large-kernel depthwise blocks, patch stem, multi-scale stages, RevIN, and the forecast head are retained; time-feature embedding and non-forecast tasks from upstream are omitted.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LuoW24,
  author       = {Donghao Luo and
                  Xue Wang},
  title        = {ModernTCN: {A} Modern Pure Convolution Structure for General Time
                  Series Analysis},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=vpJMJerXHU},
  code         = {https://github.com/luodhhh/ModernTCN},
  timestamp    = {Thu, 22 May 2025 17:54:02 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LuoW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
