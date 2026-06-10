---
model: "MTSMixer"
forecasting_setting: "time_series"
config: "configs/models/MTSMixer.toml"
registry: "models.mtsmixer.registry"
paper_title: "MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing"
venue: "arXiv preprint"
year: 2023
arxiv: "https://arxiv.org/abs/2302.04501"
---
# MTSMixer

MTSMixer is an MLP-Mixer-based model for multivariate time-series forecasting that replaces Transformer attention with two factorised mixing modules: one captures temporal dependencies and another captures cross-channel dependencies, avoiding the entanglement and redundancy introduced by joint attention. It also explicitly models the input-to-prediction mapping, yielding strong accuracy with significantly lower computational cost than Transformer-based baselines.

## Paper
- **Title**: MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-02)
- **arXiv**: https://arxiv.org/abs/2302.04501

## Abstract
Multivariate time series forecasting has been widely used in various practical scenarios. Recently, Transformer-based models have shown significant potential in forecasting tasks due to the capture of long-range dependencies. However, recent studies in the vision and NLP fields show that the role of attention modules is not clear, which can be replaced by other token aggregation operations. This paper investigates the contributions and deficiencies of attention mechanisms on the performance of time series forecasting. Specifically, we find that (1) attention is not necessary for capturing temporal dependencies, (2) the entanglement and redundancy in the capture of temporal and channel interaction affect the forecasting performance, and (3) it is important to model the mapping between the input and the prediction sequence. To this end, we propose MTS-Mixers, which use two factorized modules to capture temporal and channel dependencies. Experimental results on several real-world datasets show that MTS-Mixers outperform existing Transformer-based models with higher efficiency.

## In ModernTSF
Default config: `configs/models/MTSMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcnn/LiLRPX25,
  author       = {Zhe Li and
                  Xuanxuan Li and
                  Zhongwen Rao and
                  Lujia Pan and
                  Zenglin Xu},
  title        = {MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal
                  and Channel Mixing},
  booktitle    = {International Joint Conference on Neural Networks, {IJCNN} 2025, Rome,
                  Italy, June 30 - July 5, 2025},
  pages        = {1--8},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/IJCNN64981.2025.11229402},
  doi          = {10.1109/IJCNN64981.2025.11229402},
  timestamp    = {Fri, 21 Nov 2025 20:23:55 +0100},
  biburl       = {https://dblp.org/rec/conf/ijcnn/LiLRPX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
