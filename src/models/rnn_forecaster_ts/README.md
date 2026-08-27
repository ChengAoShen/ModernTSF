---
name: "RNNForecasterTS"
implementation: rewrite
summary: "RNNForecasterTS is a vanilla Elman RNN sequence forecaster registered for the standard time-series setting. It processes a fixed-length historical window through a single recurrent hidden layer and projects the final hidden state to the prediction horizon, providing a simple recurrent baseline for univariate and multivariate time series forecasting tasks. The ModernTSF adapter is a native PyTorch `torch.nn.Module` that runs on CPU, CUDA, or MPS accelerators via the standard trainer interface."
paper:
  title: ""
  venue: "N/A (classical baseline)"
  year: null
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# RNNForecasterTS

RNNForecasterTS is a vanilla Elman RNN sequence forecaster registered for the standard time-series setting. It processes a fixed-length historical window through a single recurrent hidden layer and projects the final hidden state to the prediction horizon, providing a simple recurrent baseline for univariate and multivariate time series forecasting tasks. The ModernTSF adapter is a native PyTorch `torch.nn.Module` that runs on CPU, CUDA, or MPS accelerators via the standard trainer interface.

<!-- model-card:canonical:start -->
## Method overview

RNNForecasterTS is a vanilla Elman RNN sequence forecaster registered for the standard time-series setting.

## Core architecture

It processes a fixed-length historical window through a single recurrent hidden layer and projects the final hidden state to the prediction horizon, providing a simple recurrent baseline for univariate and multivariate time series forecasting tasks. The ModernTSF adapter is a native PyTorch `torch.nn.Module` that runs on CPU, CUDA, or MPS accelerators via the standard trainer interface.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: not available; venue/year: N/A (classical baseline) / not available
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/RNNForecasterTS.toml`](../../../configs/models/RNNForecasterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.0`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
A vanilla (Elman) Recurrent Neural Network (RNN) consists of a single recurrent layer in which each hidden unit receives the current input and the previous hidden state, learning to summarize sequential history through a shared weight matrix. At each timestep the hidden state is updated as h_t = tanh(W_h * h_{t-1} + W_x * x_t + b), and the final hidden state is projected linearly to produce multi-step forecasts. While simple RNNs suffer from vanishing gradients over long horizons — motivating gated variants such as LSTM and GRU — they remain a useful baseline that is fast to train and easy to interpret. In ModernTSF this model is applied independently per channel (channel-independent mode) and can be accelerated on GPU/MPS via standard PyTorch tensor migration.

## In ModernTSF
Default config: `configs/models/RNNForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
