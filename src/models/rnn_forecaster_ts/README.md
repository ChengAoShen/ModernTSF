---
model: "RNNForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/RNNForecasterTS.toml"
registry: "models.rnn_forecaster_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# RNNForecasterTS

RNNForecasterTS is a vanilla Elman RNN sequence forecaster registered for the standard time-series setting. It processes a fixed-length historical window through a single recurrent hidden layer and projects the final hidden state to the prediction horizon, providing a simple recurrent baseline for univariate and multivariate time series forecasting tasks. The ModernTSF adapter is a native PyTorch `torch.nn.Module` that runs on CPU, CUDA, or MPS accelerators via the standard trainer interface.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
A vanilla (Elman) Recurrent Neural Network (RNN) consists of a single recurrent layer in which each hidden unit receives the current input and the previous hidden state, learning to summarize sequential history through a shared weight matrix. At each timestep the hidden state is updated as h_t = tanh(W_h * h_{t-1} + W_x * x_t + b), and the final hidden state is projected linearly to produce multi-step forecasts. While simple RNNs suffer from vanishing gradients over long horizons — motivating gated variants such as LSTM and GRU — they remain a useful baseline that is fast to train and easy to interpret. In ModernTSF this model is applied independently per channel (channel-independent mode) and can be accelerated on GPU/MPS via standard PyTorch tensor migration.

## In ModernTSF
Default config: `configs/models/RNNForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
