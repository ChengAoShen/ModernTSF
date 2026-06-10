---
model: "LSTMForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/LSTMForecasterTS.toml"
registry: "models.lstm_forecaster_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# LSTMForecasterTS

LSTMForecasterTS is a time series forecasting model that wraps a standard Long Short-Term Memory (LSTM) recurrent network as a direct sequence-to-sequence forecaster for univariate or multivariate time series. It is registered as a PyTorch-native adapter in ModernTSF, runs on CPU/CUDA/MPS through the standard trainer, and optionally applies RevIN (reversible instance normalisation) to handle distribution shifts.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Long Short-Term Memory (LSTM) is a gated recurrent neural network architecture introduced by Hochreiter and Schmidhuber (1997) to address the vanishing-gradient problem in standard RNNs. An LSTM cell maintains a cell state and three learned gates — input, forget, and output — that regulate how information flows across time steps, allowing the network to selectively remember or discard information over long sequences. In the forecasting setting used here, the encoder processes the historical window token-by-token and the final hidden state seeds a linear projection head that produces the full prediction horizon in one shot. No single canonical paper defines the forecasting-adapter variant; the classical LSTM architecture is the sole methodological contribution.

## In ModernTSF
Default config: `configs/models/LSTMForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
