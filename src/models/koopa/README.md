---
model: "Koopa"
forecasting_setting: "time_series"
config: "configs/models/Koopa.toml"
registry: "models.koopa.registry"
paper_title: "Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors"
venue: "NeurIPS 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2305.18803"
---
# Koopa

Koopa is a time series forecasting model for univariate and multivariate sequence prediction. It leverages modern Koopman theory to disentangle time-variant and time-invariant components of non-stationary time series, using a Fourier filter for decomposition and stackable Koopman Predictor blocks that advance each type of dynamics forward with learned linear operators.

## Paper
- **Title**: Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors
- **Venue**: NeurIPS 2023
- **Published**: 2023 (arXiv: 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.18803

## Abstract
Real-world time series are characterized by intrinsic non-stationarity that poses a principal challenge for deep forecasting models. While previous models suffer from complicated series variations induced by changing temporal distribution, we tackle non-stationary time series with modern Koopman theory that fundamentally considers the underlying time-variant dynamics. Inspired by Koopman theory of portraying complex dynamical systems, we disentangle time-variant and time-invariant components from intricate non-stationary series by Fourier Filter and design Koopman Predictor to advance respective dynamics forward. Technically, we propose Koopa as a novel Koopman forecaster composed of stackable blocks that learn hierarchical dynamics. Koopa seeks measurement functions for Koopman embedding and utilizes Koopman operators as linear portraits of implicit transition. To cope with time-variant dynamics that exhibits strong locality, Koopa calculates context-aware operators in the temporal neighborhood and is able to utilize incoming ground truth to scale up forecast horizon. Besides, by integrating Koopman Predictors into deep residual structure, we ravel out the binding reconstruction loss in previous Koopman forecasters and achieve end-to-end forecasting objective optimization. Compared with the state-of-the-art model, Koopa achieves competitive performance while saving 77.3% training time and 76.0% memory.

## In ModernTSF
Default config: `configs/models/Koopa.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
