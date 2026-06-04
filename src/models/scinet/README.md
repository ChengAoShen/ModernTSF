---
model: "SCINet"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/SCINet.toml"
registry: "models.scinet.registry"
paper_title: "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction"
venue: "NeurIPS 2022"
year: 2022
arxiv: "https://arxiv.org/abs/2106.09305"
---
# SCINet

SCINet is a CNN-based time-series forecasting model that exploits the property that temporal relations are largely preserved after downsampling. It uses a recursive downsample-convolve-interact architecture in which each layer splits the sequence into two sub-sequences, applies distinct convolutional filters to each, and then fuses them — enabling multi-resolution temporal feature extraction at O(L) complexity.

## Paper
- **Title**: SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction
- **Venue**: NeurIPS 2022
- **Published**: 2022 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.09305

## Abstract
One unique property of time series is that the temporal relations are largely preserved after downsampling into two sub-sequences. By taking advantage of this property, we propose a novel neural network architecture that conducts sample convolution and interaction for temporal modeling and forecasting, named SCINet. Specifically, SCINet is a recursive downsample-convolve-interact architecture. In each layer, we use multiple convolutional filters to extract distinct yet valuable temporal features from the downsampled sub-sequences or features. By combining these rich features aggregated from multiple resolutions, SCINet effectively models time series with complex temporal dynamics. Experimental results show that SCINet achieves significant forecasting accuracy improvements over both existing convolutional models and Transformer-based solutions across various real-world time series forecasting datasets. Our codes and data are available at https://github.com/cure-lab/SCINet.

## In ModernTSF
Default config: `configs/models/SCINet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
