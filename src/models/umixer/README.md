---
model: "UMixer"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/UMixer.toml"
registry: "models.umixer.registry"
paper_title: "U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting"
venue: "AAAI 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2401.02236"
---
# UMixer

UMixer is a long-term time-series forecasting model published at AAAI 2024. It combines U-Net-style multi-scale skip connections with MLP-Mixer blocks to capture local temporal dependencies across patches and channels separately, and introduces a stationarity correction method that explicitly restores the non-stationary distribution of the data by constraining the difference in stationarity between the model input and output.

## Paper
- **Title**: U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024 (arXiv: 2024-01)
- **arXiv**: https://arxiv.org/abs/2401.02236

## Abstract
Time series forecasting is a crucial task in various domains. Caused by factors such as trends, seasonality, or irregular fluctuations, time series often exhibits non-stationary. It obstructs stable feature propagation through deep layers, disrupts feature distributions, and complicates learning data distribution changes. As a result, many existing models struggle to capture the underlying patterns, leading to degraded forecasting performance. In this study, we tackle the challenge of non-stationarity in time series forecasting with our proposed framework called U-Mixer. By combining Unet and Mixer, U-Mixer effectively captures local temporal dependencies between different patches and channels separately to avoid the influence of distribution variations among channels, and merge low- and high-levels features to obtain comprehensive data representations. The key contribution is a novel stationarity correction method, explicitly restoring data distribution by constraining the difference in stationarity between the data before and after model processing to restore the non-stationarity information, while ensuring the temporal dependencies are preserved. Through extensive experiments on various real-world time series datasets, U-Mixer demonstrates its effectiveness and robustness, and achieves 14.5% and 7.7% improvements over state-of-the-art (SOTA) methods.

## In ModernTSF
Default config: `configs/models/UMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
