---
model: "DUET"
forecasting_setting: "time_series"
config: "configs/models/DUET.toml"
registry: "models.duet.registry"
paper_title: "DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting"
venue: "KDD 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2412.10859"
---
# DUET

DUET (Dual Clustering Enhanced Multivariate Time Series Forecasting) is a time series forecasting model that addresses two key challenges in multivariate forecasting: heterogeneous temporal patterns caused by distribution shifts, and complex inter-channel correlations. It introduces a Temporal Clustering Module (TCM) that groups time segments into fine-grained distribution clusters and assigns specialised pattern extractors to each, and a Channel Clustering Module (CCM) that performs soft channel clustering in the frequency domain via metric learning and sparsification, jointly modelling both temporal and channel dimensions.

## Paper
- **Title**: DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting
- **Venue**: KDD 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2412.10859

## Abstract
Multivariate time series forecasting is crucial for various applications, such as financial investment, energy management, weather forecasting, and traffic optimization. However, accurate forecasting is challenging due to two main factors. First, real-world time series often show heterogeneous temporal patterns caused by distribution shifts over time. Second, correlations among channels are complex and intertwined, making it hard to model the interactions among channels precisely and flexibly. In this study, we address these challenges by proposing a general framework called DUET, which introduces dual clustering on the temporal and channel dimensions to enhance multivariate time series forecasting. First, we design a Temporal Clustering Module (TCM) that clusters time series into fine-grained distributions to handle heterogeneous temporal patterns. For different distribution clusters, we design various pattern extractors to capture their intrinsic temporal patterns, thus modeling the heterogeneity. Second, we introduce a novel Channel-Soft-Clustering strategy and design a Channel Clustering Module (CCM), which captures the relationships among channels in the frequency domain through metric learning and applies sparsification to mitigate the adverse effects of noisy channels. Finally, DUET combines TCM and CCM to incorporate both the temporal and channel dimensions. Extensive experiments on 25 real-world datasets from 10 application domains, demonstrate the state-of-the-art performance of DUET.

## In ModernTSF
Default config: `configs/models/DUET.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/QiuW0GH025,
  author       = {Xiangfei Qiu and
                  Xingjian Wu and
                  Yan Lin and
                  Chenjuan Guo and
                  Jilin Hu and
                  Bin Yang},
  editor       = {Yizhou Sun and
                  Flavio Chierichetti and
                  Hady W. Lauw and
                  Claudia Perlich and
                  Wee Hyong Tok and
                  Andrew Tomkins},
  title        = {{DUET:} Dual Clustering Enhanced Multivariate Time Series Forecasting},
  booktitle    = {Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, V.1, {KDD} 2025, Toronto, ON, Canada, August 3-7,
                  2025},
  pages        = {1185--1196},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3690624.3709325},
  doi          = {10.1145/3690624.3709325},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/QiuW0GH025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
