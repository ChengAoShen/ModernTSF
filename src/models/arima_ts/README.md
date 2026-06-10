---
model: "ARIMATS"
forecasting_setting: "time_series"
config: "configs/models/ARIMATS.toml"
registry: "models.arima_ts.registry"
paper_title: "Time Series Analysis: Forecasting and Control"
venue: "Holden-Day (book) / N/A (classical baseline)"
year: 1970
arxiv: ""
---
# ARIMATS

ARIMATS is a PyTorch-native adapter for the classical ARIMA (Autoregressive Integrated Moving Average) family of statistical models, serving the standard time-series forecasting setting. It wraps differentiable ARIMA-inspired predictors — which estimate future values from differenced historical observations — inside the unified `torch.nn.Module` interface, enabling evaluation on the same trainer and benchmarking pipeline as deep learning models.

## Paper
- **Title**: Time Series Analysis: Forecasting and Control
- **Venue**: Holden-Day (book); N/A (classical baseline)
- **Published**: 1970
- **arXiv**: N/A

## Abstract
ARIMA (Autoregressive Integrated Moving Average) is a classical statistical framework for modeling and forecasting univariate time series, introduced by Box and Jenkins (1970). An ARIMA(p,d,q) model combines autoregressive terms (AR), differencing to achieve stationarity (I), and moving-average terms (MA). The model captures linear temporal dependencies by regressing the current value on its own past values and on past forecast errors, after applying d rounds of differencing to remove trend non-stationarity. Model orders (p, d, q) are typically selected via the ACF/PACF plots and information criteria such as AIC/BIC. ARIMA remains a widely used baseline for short- and medium-term forecasting across economics, meteorology, and engineering.

## In ModernTSF
Default config: `configs/models/ARIMATS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
