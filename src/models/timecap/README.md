---
name: "TimeCAP"
implementation: rewrite
summary: "TimeCAP is a time series forecasting model for multivariate sequence prediction. It is the first purely channel-aware pre-training framework for multivariate time series, systematically integrating complementary autoregressive and one-shot generative paradigms via a flexible channel-grouping learning approach and an adaptive meta-routing mechanism that captures both intra-group local patterns and global inter-channel coherence."
paper:
  title: "TimeCAP: A Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting"
  venue: "AAAI 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# TimeCAP

TimeCAP is a time series forecasting model for multivariate sequence prediction. It is the first purely channel-aware pre-training framework for multivariate time series, systematically integrating complementary autoregressive and one-shot generative paradigms via a flexible channel-grouping learning approach and an adaptive meta-routing mechanism that captures both intra-group local patterns and global inter-channel coherence.

<!-- model-card:canonical:start -->
## Method overview

TimeCAP is a time series forecasting model for multivariate sequence prediction.

## Core architecture

It is the first purely channel-aware pre-training framework for multivariate time series, systematically integrating complementary autoregressive and one-shot generative paradigms via a flexible channel-grouping learning approach and an adaptive meta-routing mechanism that captures both intra-group local patterns and global inter-channel coherence.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: TimeCAP: A Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting; venue/year: AAAI 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TimeCAP.toml`](../../../configs/models/TimeCAP.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeCAP: A Channel-Aware Pre-Training Framework for Multivariate Time Series Forecasting
- **Venue**: AAAI 2026 (Oral)
- **Published**: 2026
- **arXiv**: N/A

## Abstract
TimeCAP introduces the first purely channel-aware pre-training framework for multivariate time series, internalizing latent causal relationships among variables inherent in multi-domain data and effectively transferring the acquired knowledge to downstream applications. Existing approaches exhibit two critical limitations: underestimating the significance of multivariate dependencies in learning generalizable representations, and failing to reconcile the complementary strengths of autoregressive and one-shot generative paradigms. TimeCAP addresses both by presenting a flexible channel-grouping learning approach, complemented by an adaptive meta-routing mechanism, enabling the model to simultaneously recognize intra-group local patterns while maintaining global coherence. Intra- and inter-group multivariate dependencies are captured through self- and cross-attention with a channel-aware mask, which strictly confines interactions among time-aligned, fine-grained multivariate tokens. In few-shot evaluation, TimeCAP achieves average MSE and MAE reductions of 11.8% and 6% over leading baselines, while also outperforming state-of-the-art models in full-shot and zero-shot settings by large margins.

## In ModernTSF
Default config: `configs/models/TimeCAP.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/RenLHZZLLL26,
  author       = {Chuanru Ren and
                  Yao Lu and
                  Tianjin Huang and
                  Haowen Zheng and
                  Hengde Zhu and
                  Yunyin Li and
                  Hengxiao Li and
                  Lu Liu},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {TimeCAP: {A} Channel-Aware Pre-Training Framework for Multivariate
                  Time Series Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25108--25116},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39700},
  doi          = {10.1609/AAAI.V40I30.39700},
  timestamp    = {Fri, 27 Mar 2026 07:38:55 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/RenLHZZLLL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
