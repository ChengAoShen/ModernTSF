---
model: "RLinear"
forecasting_setting: "time_series"
config: "configs/models/RLinear.toml"
registry: "models.rlinear.registry"
paper_title: "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping"
venue: "arXiv preprint"
year: 2023
arxiv: "https://arxiv.org/abs/2305.10721"
---
# RLinear

RLinear is a time series forecasting model that combines Reversible Instance Normalisation (RevIN) with a single linear projection layer to perform long-term multivariate or univariate forecasting. Despite its simplicity, the model achieves competitive or state-of-the-art performance on standard benchmarks by exploiting the fact that affine mapping dominates forecasting accuracy and that RevIN transforms non-periodic trends into periodic-like patterns that a linear layer can capture effectively.

## Paper
- **Title**: Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-05)
- **arXiv**: https://arxiv.org/abs/2305.10721

## Abstract
Long-term time series forecasting (LTSF) has gained significant attention in recent years. While various specialized designs exist for capturing temporal dependency, recent studies have shown that even a single linear layer can achieve competitive performance. This paper investigates the intrinsic effectiveness of recent LTSF approaches and reveals the critical role of affine mapping. We conduct comprehensive experiments on both simulated and real-world datasets to analyze the components of state-of-the-art models. A theoretical analysis is provided to explain the working mechanisms of affine mapping in periodic signal forecasting. We evaluate the impact of reversible normalization and input horizon extension on model robustness. We find that (1) affine mapping dominates forecasting performance across commonly utilized benchmarks, with models learning similar transition matrices from input to output; (2) affine mapping effectively captures periodic patterns but struggles with non-periodic signals or time series with varying periods across channels; (3) reversible normalization significantly enhances trend forecasting by transforming non-periodic trends into periodic-like patterns; (4) increasing input horizon improves performance on multi-channel data with different periods. Code is available at: https://github.com/plumprc/RTSF. Our findings provide theoretical and experimental insights into the working mechanisms of LTSF models, highlighting both the strengths and limitations of linear approaches. The results suggest that future model development should focus on handling cross-channel period variations and non-periodic components.

## In ModernTSF
Default config: `configs/models/RLinear.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2305-10721,
  author       = {Zhe Li and
                  Shiyi Qi and
                  Yiduo Li and
                  Zenglin Xu},
  title        = {Revisiting Long-term Time Series Forecasting: An Investigation on
                  Linear Mapping},
  journal      = {CoRR},
  volume       = {abs/2305.10721},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.10721},
  doi          = {10.48550/ARXIV.2305.10721},
  eprinttype   = {arXiv},
  eprint       = {2305.10721},
  timestamp    = {Thu, 25 May 2023 15:41:47 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-10721.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
