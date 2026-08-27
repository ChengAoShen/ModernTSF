---
name: "RNNForecasterTS"
implementation: rewrite
summary: "RNNForecasterTS is a clean-room Elman RNN baseline that encodes a fixed history and directly projects the final hidden state to a multistep forecast."
paper:
  title: "Finding Structure in Time"
  venue: "Cognitive Science"
  year: 1990
  url: "https://doi.org/10.1207/s15516709cog1402_1"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# RNNForecasterTS

RNNForecasterTS is a clean-room Elman RNN baseline that encodes a fixed history and directly projects the final hidden state to a multistep forecast.

<!-- model-card:canonical:start -->
## Method overview

RNNForecasterTS is a clean-room Elman RNN baseline that encodes a fixed history and directly projects the final hidden state to a multistep forecast.

## Core architecture

RNNForecasterTS is a clean-room Elman RNN baseline that encodes a fixed history and directly projects the final hidden state to a multistep forecast.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://doi.org/10.1207/s15516709cog1402_1); title: Finding Structure in Time; venue/year: Cognitive Science / 1990
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/RNNForecasterTS.toml`](../../../configs/models/RNNForecasterTS.toml).

## Differences

Clean-room implementation: confirmed. The local code was independently designed from Elman's published recurrence and the repository tensor contract; no external implementation source was copied. Elman (1990) does not define the direct multi-horizon head, RevIN, or this joint multivariate forecasting setup, so no experimental parity is claimed. Formula and full runtime-contract evidence are recorded in `verification/rewrite/RNNForecasterTS.json`.

## Shared components

- [`revin`](../../components/revin.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.0`, `num_layers=1`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Finding Structure in Time
- **Venue**: Cognitive Science
- **Published**: 1990
- **DOI**: https://doi.org/10.1207/s15516709cog1402_1

## Abstract
A vanilla (Elman) Recurrent Neural Network (RNN) consists of a recurrent layer in which each hidden unit receives the current multivariate input and previous hidden state. At each timestep the hidden state is updated as h_t = tanh(W_h h_{t-1} + W_x x_t + b), and the final hidden state is projected linearly to produce the complete multi-step, multi-channel forecast. Simple RNNs can suffer from vanishing gradients over long histories, which motivates gated variants such as LSTM and GRU.

## Source and verification

Clean-room implementation: confirmed. The local code was independently designed from Elman's published recurrence and the repository tensor contract; no external implementation source was copied. Elman (1990) does not define the direct multi-horizon head, RevIN, or this joint multivariate forecasting setup, so no experimental parity is claimed. Formula and full runtime-contract evidence are recorded in `verification/rewrite/RNNForecasterTS.json`.

## In ModernTSF
Default config: `configs/models/RNNForecasterTS.toml`; model specification: `spec.py`; clean-room implementation: `model.py`.

## Citation

```bibtex
@article{elman1990finding,
  author  = {Jeffrey L. Elman},
  title   = {Finding Structure in Time},
  journal = {Cognitive Science},
  volume  = {14},
  number  = {2},
  pages   = {179--211},
  year    = {1990},
  doi     = {10.1207/s15516709cog1402_1}
}
```
