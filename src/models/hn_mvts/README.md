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

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/SavchenkoK26,
  author       = {Andrey V. Savchenko and
                  Oleg Kachan},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {{HN-MVTS:} HyperNetwork-based Multivariate Time Series Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25200--25208},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39711},
  doi          = {10.1609/AAAI.V40I30.39711},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/SavchenkoK26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
