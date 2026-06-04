---
model: "HN_MVTS"
forecasting_setting: "time_series"
config: "configs/models/HN_MVTS.toml"
registry: "models.hn_mvts.registry"
paper_title: "HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2511.08340"
---
# HN_MVTS

HN_MVTS integrates a hypernetwork-based generative prior with any base neural-network forecaster for multivariate time-series forecasting. The hypernetwork takes a learnable embedding matrix of time-series components as input and generates the weights of the base model's final layer, acting as a data-adaptive regulariser that improves generalisation and long-range predictive accuracy — used only during training so it adds no inference overhead. This approach bridges the gap between high-accuracy channel-dependent models and the robustness of channel-independent models.

## Paper
- **Title**: HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-11)
- **arXiv**: https://arxiv.org/abs/2511.08340

## Abstract
Accurate forecasting of multivariate time series data remains a formidable challenge, particularly due to the growing complexity of temporal dependencies in real-world scenarios. While neural network-based models have achieved notable success in this domain, complex channel-dependent models often suffer from performance degradation compared to channel-independent models that do not consider the relationship between components but provide high robustness due to small capacity. In this work, we propose HN-MVTS, a novel architecture that integrates a hypernetwork-based generative prior with an arbitrary neural network forecasting model. The input of this hypernetwork is a learnable embedding matrix of time series components. To restrict the number of new parameters, the hypernetwork learns to generate the weights of the last layer of the target forecasting networks, serving as a data-adaptive regularizer that improves generalization and long-range predictive accuracy. The hypernetwork is used only during the training, so it does not increase the inference time compared to the base forecasting model. Extensive experiments on eight benchmark datasets demonstrate that application of HN-MVTS to the state-of-the-art models (DLinear, PatchTST, TSMixer, etc.) typically improves their performance. Our findings suggest that hypernetwork-driven parameterization offers a promising direction for enhancing existing forecasting techniques in complex scenarios.

## In ModernTSF
Default config: `configs/models/HN_MVTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
