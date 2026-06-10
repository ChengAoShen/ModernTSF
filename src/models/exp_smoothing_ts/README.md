---
model: "ExpSmoothingTS"
forecasting_setting: "time_series"
config: "configs/models/ExpSmoothingTS.toml"
registry: "models.exp_smoothing_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# ExpSmoothingTS

ExpSmoothingTS is a PyTorch-native time series forecasting adapter that implements an exponential-smoothing-inspired predictor for the standard time series forecasting setting. It uses learned decay weights to progressively downweight older observations, extrapolates trends from the smoothed history, and runs through the ModernTSF standard trainer so it can be evaluated on GPU/CPU alongside deep learning models.

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Exponential smoothing is a classical family of time series forecasting methods that assign exponentially decreasing weights to past observations, placing the most emphasis on recent data. Simple exponential smoothing forecasts a constant level, while double (Holt) and triple (Holt-Winters) variants additionally model additive or multiplicative trend and seasonality components via additional smoothing parameters. The ExpSmoothingTS adapter in ModernTSF re-implements the core smoothing idea as a differentiable PyTorch module with learnable decay parameters, enabling the classical technique to be trained end-to-end with gradient descent and deployed on the same hardware as neural forecasting models.

## In ModernTSF
Default config: `configs/models/ExpSmoothingTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{holt2004forecasting,
  author  = {Charles C. Holt},
  title   = {Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages},
  journal = {International Journal of Forecasting},
  volume  = {20},
  number  = {1},
  pages   = {5--10},
  year    = {2004},
  doi     = {10.1016/j.ijforecast.2003.09.015}
}
```
