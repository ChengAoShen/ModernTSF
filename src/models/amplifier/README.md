---
model: "Amplifier"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/Amplifier.toml"
registry: "models.amplifier.registry"
paper_title: "Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2501.17216"
---
# Amplifier

Amplifier is a multivariate/univariate time-series forecasting model that addresses the common failure mode of existing models that overlook low-energy frequency components. It introduces an energy amplification technique — comprising an amplification block and a restoration block — integrated with a seasonal-trend decomposition backbone, and further augments it with a semi-channel interaction temporal relationship enhancement block that exploits both commonality and specificity across channels.

## Paper
- **Title**: Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2025-01)
- **arXiv**: https://arxiv.org/abs/2501.17216

## Abstract
We propose an energy amplification technique to address the issue that existing models easily overlook low-energy components in time series forecasting. This technique comprises an energy amplification block and an energy restoration block. The energy amplification block enhances the energy of low-energy components to improve the model's learning efficiency for these components, while the energy restoration block returns the energy to its original level. Moreover, considering that the energy-amplified data typically displays two distinct energy peaks in the frequency spectrum, we integrate the energy amplification technique with a seasonal-trend forecaster to model the temporal relationships of these two peaks independently, serving as the backbone for our proposed model, Amplifier. Additionally, we propose a semi-channel interaction temporal relationship enhancement block for Amplifier, which enhances the model's ability to capture temporal relationships from the perspective of the commonality and specificity of each channel in the data. Extensive experiments on eight time series forecasting benchmarks consistently demonstrate our model's superiority in both effectiveness and efficiency compared to state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/Amplifier.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
