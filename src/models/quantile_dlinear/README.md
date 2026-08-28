---
name: "QuantileDLinear"
implementation: rewrite
summary: "QuantileDLinear is a **probabilistic** ModernTSF forecaster: it wraps the point DLinear backbone with the shared monotone `QuantileHead` (`src/components/quantile_head.py`) to emit a non-crossing grid of quantiles `(B, pred_len, C, Q)` instead of a single point. The head builds quantiles from a median anchor by adding/subtracting cumulative `softplus` offsets, so the predicted quantiles cannot cross by construction. It is trained with the pinball (`quantile`) loss and scored with CRPS / WQL / coverage."
paper:
  title: "Are Transformers Effective for Time Series Forecasting? (DLinear backbone)"
  venue: "AAAI 2023"
  year: 2023
  url: "https://arxiv.org/abs/2205.13504"
codebase:
  url: "https://github.com/cure-lab/LTSF-Linear"
  revision: "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
  license: "Apache-2.0"
  usage: reference-only
---
# QuantileDLinear

QuantileDLinear is a **probabilistic** ModernTSF forecaster: it wraps the point
DLinear backbone with the shared monotone `QuantileHead`
(`src/components/quantile_head.py`) to emit a non-crossing grid of quantiles
`(B, pred_len, C, Q)` instead of a single point. The head builds quantiles from a
median anchor by adding/subtracting cumulative `softplus` offsets, so the
predicted quantiles cannot cross by construction. It is trained with the pinball
(`quantile`) loss and scored with CRPS / WQL / coverage.

<!-- model-card:canonical:start -->
## Method overview

QuantileDLinear is a **probabilistic** ModernTSF forecaster: it wraps the point DLinear backbone with the shared monotone `QuantileHead` (`src/components/quantile_head.py`) to emit a non-crossing grid of quantiles `(B, pred_len, C, Q)` instead of a single point.

## Core architecture

The head builds quantiles from a median anchor by adding/subtracting cumulative `softplus` offsets, so the predicted quantiles cannot cross by construction. It is trained with the pinball (`quantile`) loss and scored with CRPS / WQL / coverage.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, quantiles]` quantile forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2205.13504); title: Are Transformers Effective for Time Series Forecasting? (DLinear backbone); venue/year: AAAI 2023 / 2023
- [codebase](https://github.com/cure-lab/LTSF-Linear); revision: `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`; license: `Apache-2.0`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/QuantileDLinear.toml`](../../../configs/models/QuantileDLinear.toml).

## Differences

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independently composed from verified shared components; no upstream source was copied.
- The probabilistic monotone head and pinball-loss protocol are ModernTSF additions; this is not a model or result claimed by the DLinear paper.

## Shared components

- [`dlinear`](../../components/dlinear.py)
- [`quantile_head`](../../components/quantile_head.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `kernel_size=25`, `individual=False`
<!-- model-card:canonical:end -->

## Method
- **Backbone**: DLinear — trend + seasonal decomposition with two linear heads
  (Zeng et al., AAAI 2023, arXiv: 2205.13504).
- **Probabilistic head**: monotone quantile regression (pinball loss; Koenker &
  Bassett, 1978).

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/QuantileDLinear.toml`; specification: `spec.py`; implementation:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

Clean-room implementation: confirmed. Reference-only source code was not copied.

- Independently composed from verified shared components; no upstream source was copied.
- The probabilistic monotone head and pinball-loss protocol are ModernTSF additions; this is not a model or result claimed by the DLinear paper.
