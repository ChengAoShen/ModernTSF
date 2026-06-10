---
model: "Autoformer"
forecasting_setting: "time_series"
config: "configs/models/Autoformer.toml"
registry: "models.autoformer.registry"
paper_title: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
venue: "NeurIPS 2021"
year: 2021
arxiv: "https://arxiv.org/abs/2106.13008"
---
# Autoformer

Autoformer is a Transformer-based model for long-term multivariate time series forecasting that replaces the standard self-attention mechanism with an Auto-Correlation mechanism and incorporates a progressive series decomposition block as a core inner component of the deep network rather than a pre-processing step.

## Paper
- **Title**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
- **Venue**: NeurIPS 2021
- **Published**: 2021 (arXiv: 2021-06)
- **arXiv**: https://arxiv.org/abs/2106.13008

## Abstract
Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease.

## In ModernTSF
Default config: `configs/models/Autoformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/WuXWL21,
  author       = {Haixu Wu and
                  Jiehui Xu and
                  Jianmin Wang and
                  Mingsheng Long},
  editor       = {Marc'Aurelio Ranzato and
                  Alina Beygelzimer and
                  Yann N. Dauphin and
                  Percy Liang and
                  Jennifer Wortman Vaughan},
  title        = {Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term
                  Series Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 34: Annual Conference
                  on Neural Information Processing Systems 2021, NeurIPS 2021, December
                  6-14, 2021, virtual},
  pages        = {22419--22430},
  year         = {2021},
  url          = {https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html},
  timestamp    = {Mon, 26 Jun 2023 20:41:56 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/WuXWL21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
