---
model: "NBeats"
forecasting_setting: "time_series"
config: "configs/models/NBeats.toml"
registry: "models.nbeats.registry"
paper_title: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
venue: "ICLR 2020"
year: 2020
arxiv: "https://arxiv.org/abs/1905.10437"
---
# NBeats

N-BEATS is a deep neural architecture for univariate and multivariate time series point forecasting. It is built on a deep stack of fully-connected blocks, each producing a backcast (reconstruction of the input) and a forecast, linked by backward and forward residual connections. Two configurations exist: a generic version relying solely on deep learning primitives, and an interpretable version whose basis expansion functions correspond to trend and seasonality components.

## Paper
- **Title**: N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
- **Venue**: ICLR 2020
- **Published**: 2020 (arXiv: 2019-05)
- **arXiv**: https://arxiv.org/abs/1905.10437

## Abstract
We focus on solving the univariate times series point forecasting problem using deep learning. We propose a deep neural architecture based on backward and forward residual links and a very deep stack of fully-connected layers. The architecture has a number of desirable properties, being interpretable, applicable without modification to a wide array of target domains, and fast to train. We test the proposed architecture on several well-known datasets, including M3, M4 and TOURISM competition datasets containing time series from diverse domains. We demonstrate state-of-the-art performance for two configurations of N-BEATS for all the datasets, improving forecast accuracy by 11% over a statistical benchmark and by 3% over last year's winner of the M4 competition, a domain-adjusted hand-crafted hybrid between neural network and statistical time series models. The first configuration of our model does not employ any time-series-specific components and its performance on heterogeneous datasets strongly suggests that, contrarily to received wisdom, deep learning primitives such as residual blocks are by themselves sufficient to solve a wide range of forecasting problems. Finally, we demonstrate how the proposed architecture can be augmented to provide outputs that are interpretable without considerable loss in accuracy.

## In ModernTSF
Default config: `configs/models/NBeats.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
