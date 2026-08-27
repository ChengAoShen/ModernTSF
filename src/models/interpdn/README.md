---
name: "InterPDN"
implementation: rewrite
summary: "InterPDN (interleaved dual-branch Probability Distribution Network) is a time series forecasting model for standard multivariate or univariate sequences. Rather than predicting a scalar at each future step, it directly constructs a discrete probability distribution per step; the regression output is computed as the expectation over a predefined support set. A dual-branch architecture with interleaved support sets, coarse temporal-scale branches for long-term trend, and self-supervised consistency constraints between branches further improves robustness."
paper:
  title: "Time Series Forecasting via Direct Per-Step Probability Distribution Modeling"
  venue: "AAAI 2026"
  year: 2026
  url: "https://arxiv.org/abs/2511.23260"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# InterPDN

InterPDN (interleaved dual-branch Probability Distribution Network) is a time series forecasting model for standard multivariate or univariate sequences. Rather than predicting a scalar at each future step, it directly constructs a discrete probability distribution per step; the regression output is computed as the expectation over a predefined support set. A dual-branch architecture with interleaved support sets, coarse temporal-scale branches for long-term trend, and self-supervised consistency constraints between branches further improves robustness.

<!-- model-card:canonical:start -->
## Method overview

InterPDN (interleaved dual-branch Probability Distribution Network) is a time series forecasting model for standard multivariate or univariate sequences.

## Core architecture

Rather than predicting a scalar at each future step, it directly constructs a discrete probability distribution per step; the regression output is computed as the expectation over a predefined support set. A dual-branch architecture with interleaved support sets, coarse temporal-scale branches for long-term trend, and self-supervised consistency constraints between branches further improves robustness.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2511.23260); title: Time Series Forecasting via Direct Per-Step Probability Distribution Modeling; venue/year: AAAI 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/InterPDN.toml`](../../../configs/models/InterPDN.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Time Series Forecasting via Direct Per-Step Probability Distribution Modeling
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-11)
- **arXiv**: https://arxiv.org/abs/2511.23260

## Abstract
Deep neural network-based time series prediction models have recently demonstrated superior capabilities in capturing complex temporal dependencies. However, it is challenging for these models to account for uncertainty associated with their predictions, because they directly output scalar values at each time step. To address such a challenge, we propose a novel model named interleaved dual-branch Probability Distribution Network (interPDN), which directly constructs discrete probability distributions per step instead of a scalar. The regression output at each time step is derived by computing the expectation of the predictive distribution on a predefined support set. To mitigate prediction anomalies, a dual-branch architecture is introduced with interleaved support sets, augmented by coarse temporal-scale branches for long-term trend forecasting. Outputs from another branch are treated as auxiliary signals to impose self-supervised consistency constraints on the current branch's prediction. Extensive experiments on multiple real-world datasets demonstrate the superior performance of interPDN.

## In ModernTSF
Default config: `configs/models/InterPDN.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/KongH26,
  author       = {Linghao Kong and
                  Xiaopeng Hong},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Time Series Forecasting via Direct Per-Step Probability Distribution
                  Modeling},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {22653--22661},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i27.39426},
  doi          = {10.1609/AAAI.V40I27.39426},
  timestamp    = {Thu, 26 Mar 2026 16:46:49 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/KongH26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
