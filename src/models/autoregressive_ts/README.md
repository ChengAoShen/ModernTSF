---
model: "AutoRegressiveTS"
forecasting_setting: "time_series"
config: "configs/models/AutoRegressiveTS.toml"
registry: "models.autoregressive_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# AutoRegressiveTS

AutoRegressiveTS is a classical autoregressive lag model for univariate and multivariate time-series forecasting. It directly maps the historical input window to the future prediction window using a learned linear projection over lagged observations, and is wrapped as a PyTorch `nn.Module` so that it integrates with the standard ModernTSF training loop and can run on CUDA/MPS devices.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Autoregressive (AR) models predict the next value (or block of values) in a time series as a linear combination of a fixed number of past observations, known as the lag order. The parameters are typically estimated by ordinary least squares or Yule–Walker equations. When extended to the vector setting (VAR), each variable is regressed on its own lags and the lags of all other variables. The AR/VAR family is one of the oldest and most studied approaches in time-series analysis, forming the basis for more complex models such as ARIMA and state-space methods. In ModernTSF the model is implemented as a differentiable linear layer that maps the full input window to the full prediction horizon in a single forward pass, enabling end-to-end gradient-based training.

## In ModernTSF
Default config: `configs/models/AutoRegressiveTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@book{box1970time,
  author    = {George E. P. Box and Gwilym M. Jenkins},
  title     = {Time Series Analysis: Forecasting and Control},
  publisher = {Holden-Day},
  address   = {San Francisco},
  year      = {1970},
  url       = {https://archive.org/details/timeseriesanalys0000boxg}
}
```
