---
model: "PaiFilter"
forecasting_setting: "time_series"
config: "configs/models/PaiFilter.toml"
registry: "models.paifilter.registry"
paper_title: "FilterNet: Harnessing Frequency Filters for Time Series Forecasting"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2411.01623"
---
# PaiFilter

PaiFilter implements the plain shaping filter variant from the FilterNet framework for time series forecasting. It adopts a universal frequency kernel for signal filtering and temporal modeling, using randomly initialized learnable weight parameters that are multiplied with the input to selectively pass or attenuate frequency components. This design allows FilterNet-style forecasting without the contextual gating of the full FilterNet model, serving as an efficient baseline for frequency-domain time series forecasting.

## Paper
- **Title**: FilterNet: Harnessing Frequency Filters for Time Series Forecasting
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-11)
- **arXiv**: https://arxiv.org/abs/2411.01623

## Abstract
Given the ubiquitous presence of time series data across various domains, precise forecasting of time series holds significant importance and finds widespread real-world applications such as energy, weather, healthcare, etc. While numerous forecasters have been proposed using different network architectures, the Transformer-based models have state-of-the-art performance in time series forecasting. However, forecasters based on Transformers are still suffering from vulnerability to high-frequency signals, efficiency in computation, and bottleneck in full-spectrum utilization, which essentially are the cornerstones for accurately predicting time series with thousands of points. In this paper, we explore a novel perspective of enlightening signal processing for deep time series forecasting. Inspired by the filtering process, we introduce one simple yet effective network, namely FilterNet, built upon our proposed learnable frequency filters to extract key informative temporal patterns by selectively passing or attenuating certain components of time series signals. Concretely, we propose two kinds of learnable filters in the FilterNet: (i) Plain shaping filter, that adopts a universal frequency kernel for signal filtering and temporal modeling; (ii) Contextual shaping filter, that utilizes filtered frequencies examined in terms of its compatibility with input signals for dependency learning. Equipped with the two filters, FilterNet can approximately surrogate the linear and attention mappings widely adopted in time series literature, while enjoying superb abilities in handling high-frequency noises and utilizing the whole frequency spectrum that is beneficial for forecasting. Finally, we conduct extensive experiments on eight time series forecasting benchmarks, and experimental results have demonstrated our superior performance in terms of both effectiveness and efficiency compared with state-of-the-art methods. Our code is available at https://github.com/aikunyi/FilterNet.

## In ModernTSF
Default config: `configs/models/PaiFilter.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
