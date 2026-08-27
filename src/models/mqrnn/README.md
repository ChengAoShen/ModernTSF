---
name: "MQRNN"
implementation: rewrite
summary: "MQRNN (Multi-horizon Quantile Recurrent forecaster) is a **probabilistic** sequence-to-sequence model: an RNN encoder summarizes the input window into a context, and a global MLP decoder emits all horizon steps jointly as quantiles. In ModernTSF the decoder feeds the shared monotone `QuantileHead` (`src/components/quantile_head.py`), giving a non-crossing quantile grid `(B, pred_len, C, Q)` trained with the pinball (`quantile`) loss and scored with CRPS / WQL / coverage."
paper:
  title: "A Multi-Horizon Quantile Recurrent Forecaster"
  venue: "NeurIPS 2017 Time Series Workshop"
  year: 2017
  url: "https://arxiv.org/abs/1711.11053"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# MQRNN

MQRNN (Multi-horizon Quantile Recurrent forecaster) is a **probabilistic**
sequence-to-sequence model: an RNN encoder summarizes the input window into a
context, and a global MLP decoder emits all horizon steps jointly as quantiles.
In ModernTSF the decoder feeds the shared monotone `QuantileHead`
(`src/components/quantile_head.py`), giving a non-crossing quantile grid
`(B, pred_len, C, Q)` trained with the pinball (`quantile`) loss and scored with
CRPS / WQL / coverage.

<!-- model-card:canonical:start -->
## Method overview

MQRNN (Multi-horizon Quantile Recurrent forecaster) is a **probabilistic** sequence-to-sequence model: an RNN encoder summarizes the input window into a context, and a global MLP decoder emits all horizon steps jointly as quantiles.

## Core architecture

In ModernTSF the decoder feeds the shared monotone `QuantileHead` (`src/components/quantile_head.py`), giving a non-crossing quantile grid `(B, pred_len, C, Q)` trained with the pinball (`quantile`) loss and scored with CRPS / WQL / coverage.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels, quantiles]` quantile forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1711.11053); title: A Multi-Horizon Quantile Recurrent Forecaster; venue/year: NeurIPS 2017 Time Series Workshop / 2017
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/MQRNN.toml`](../../../configs/models/MQRNN.toml).

## Differences

- Implementation: `rewrite` (clean-room audit pending); no author implementation or pinned upstream source has been established.
- This uses a shared channel-independent GRU, joint horizon MLP, and ModernTSF monotone quantile head. It does not implement the paper's static/future-covariate global/local decoder.
- Paper protocol and result reproduction remain blocked pending a traceable reference and dataset-aligned experiment.

## Shared components

- [`quantile_head`](../../components/quantile_head.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `hidden_size=64`, `num_layers=1`, `decoder_hidden=64`, `dropout=0.1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: A Multi-Horizon Quantile Recurrent Forecaster
- **Authors**: Wen, Torkkola, Narayanaswamy, Madeka
- **Published**: 2017
- **arXiv**: https://arxiv.org/abs/1711.11053

## In ModernTSF
`output_type = "quantile"`; pair with `[training] loss = "quantile"`. Default
config: `configs/models/MQRNN.toml`; specification: `spec.py`; implementation:
`model.py`. `quantile_levels` are injected from
`evaluation.quantile_levels`. Use the model specification and probabilistic output contract.

## Source and verification

- Implementation: `rewrite` (clean-room audit pending); no author implementation or pinned upstream source has been established.
- This uses a shared channel-independent GRU, joint horizon MLP, and ModernTSF monotone quantile head. It does not implement the paper's static/future-covariate global/local decoder.
- Paper protocol and result reproduction remain blocked pending a traceable reference and dataset-aligned experiment.
