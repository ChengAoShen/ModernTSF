---
name: "LSTMForecasterTS"
implementation: rewrite
summary: "LSTMForecasterTS is a time series forecasting model that wraps a standard Long Short-Term Memory (LSTM) recurrent network as a direct sequence-to-sequence forecaster for univariate or multivariate time series. It is registered as a PyTorch-native adapter in ModernTSF, runs on CPU/CUDA/MPS through the standard trainer, and optionally applies RevIN (reversible instance normalisation) to handle distribution shifts."
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
# LSTMForecasterTS

LSTMForecasterTS is a time series forecasting model that wraps a standard Long Short-Term Memory (LSTM) recurrent network as a direct sequence-to-sequence forecaster for univariate or multivariate time series. It is registered as a PyTorch-native adapter in ModernTSF, runs on CPU/CUDA/MPS through the standard trainer, and optionally applies RevIN (reversible instance normalisation) to handle distribution shifts.

<!-- model-card:canonical:start -->
## Method overview

LSTMForecasterTS is a time series forecasting model that wraps a standard Long Short-Term Memory (LSTM) recurrent network as a direct sequence-to-sequence forecaster for univariate or multivariate time series.

## Core architecture

It is registered as a PyTorch-native adapter in ModernTSF, runs on CPU/CUDA/MPS through the standard trainer, and optionally applies RevIN (reversible instance normalisation) to handle distribution shifts.

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
[`configs/models/LSTMForecasterTS.toml`](../../../configs/models/LSTMForecasterTS.toml).

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
Long Short-Term Memory (LSTM) is a gated recurrent neural network architecture introduced by Hochreiter and Schmidhuber (1997) to address the vanishing-gradient problem in standard RNNs. An LSTM cell maintains a cell state and three learned gates — input, forget, and output — that regulate how information flows across time steps, allowing the network to selectively remember or discard information over long sequences. In the forecasting setting used here, the encoder processes the historical window token-by-token and the final hidden state seeds a linear projection head that produces the full prediction horizon in one shot. No single canonical paper defines the forecasting-adapter variant; the classical LSTM architecture is the sole methodological contribution.

## In ModernTSF
Default config: `configs/models/LSTMForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{hochreiter1997long,
  author  = {Sepp Hochreiter and J{\"u}rgen Schmidhuber},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  doi     = {10.1162/neco.1997.9.8.1735}
}
```
