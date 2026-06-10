---
model: "Fredformer"
forecasting_setting: "time_series"
config: "configs/models/Fredformer.toml"
registry: "models.fredformer.registry"
paper_title: "Fredformer: Frequency Debiased Transformer for Time Series Forecasting"
venue: "KDD 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2406.09009"
---
# Fredformer

Fredformer is a Transformer-based model for multivariate time series forecasting that addresses the frequency bias problem, where attention mechanisms tend to over-weight low-frequency (high-energy) components while ignoring high-frequency features. It applies frequency-equalised attention across patch-level frequency bands to produce more balanced and accurate forecasts.

## Paper
- **Title**: Fredformer: Frequency Debiased Transformer for Time Series Forecasting
- **Venue**: KDD 2024
- **Published**: 2024 (arXiv: 2024-06)
- **arXiv**: https://arxiv.org/abs/2406.09009

## Abstract
The Transformer model has shown leading performance in time series forecasting. Nevertheless, in some complex scenarios, it tends to learn low-frequency features in the data and overlook high-frequency features, showing a frequency bias. This bias prevents the model from accurately capturing important high-frequency data features. In this paper, we undertook empirical analyses to understand this bias and discovered that frequency bias results from the model disproportionately focusing on frequency features with higher energy. Based on our analysis, we formulate this bias and propose Fredformer, a Transformer-based framework designed to mitigate frequency bias by learning features equally across different frequency bands. This approach prevents the model from overlooking lower amplitude features important for accurate forecasting. Extensive experiments show the effectiveness of our proposed approach, which can outperform other baselines in different real-world time-series datasets. Furthermore, we introduce a lightweight variant of the Fredformer with an attention matrix approximation, which achieves comparable performance but with much fewer parameters and lower computation costs.

## In ModernTSF
Default config: `configs/models/Fredformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/Piao0MMS24,
  author       = {Xihao Piao and
                  Zheng Chen and
                  Taichi Murayama and
                  Yasuko Matsubara and
                  Yasushi Sakurai},
  editor       = {Ricardo Baeza{-}Yates and
                  Francesco Bonchi},
  title        = {Fredformer: Frequency Debiased Transformer for Time Series Forecasting},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2024, Barcelona, Spain, August 25-29, 2024},
  pages        = {2400--2410},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3637528.3671928},
  doi          = {10.1145/3637528.3671928},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/Piao0MMS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
