---
model: "TSMixer"
forecasting_setting: "time_series"
config: "configs/models/TSMixer.toml"
registry: "models.tsmixer.registry"
paper_title: "TSMixer: An All-MLP Architecture for Time Series Forecasting"
venue: "TMLR 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2303.06053"
---
# TSMixer

TSMixer is an MLP-Mixer-style model for multivariate time-series forecasting that alternates mixing operations along the time dimension and the feature (channel) dimension. By stacking MLP blocks that operate on transposed views of the input, it efficiently extracts both temporal dynamics and cross-variate correlations without any attention mechanism, achieving competitive accuracy while remaining easy to implement.

## Paper
- **Title**: TSMixer: An All-MLP Architecture for Time Series Forecasting
- **Venue**: TMLR 2023
- **Published**: 2023 (arXiv: 2023-03)
- **arXiv**: https://arxiv.org/abs/2303.06053

## Abstract
Real-world time-series datasets are often multivariate with complex dynamics. To capture this complexity, high capacity architectures like recurrent- or attention-based sequential deep learning models have become popular. However, recent work demonstrates that simple univariate linear models can outperform such deep learning models on several commonly used academic benchmarks. Extending them, in this paper, we investigate the capabilities of linear models for time-series forecasting and present Time-Series Mixer (TSMixer), a novel architecture designed by stacking multi-layer perceptrons (MLPs). TSMixer is based on mixing operations along both the time and feature dimensions to extract information efficiently. On popular academic benchmarks, the simple-to-implement TSMixer is comparable to specialized state-of-the-art models that leverage the inductive biases of specific benchmarks. On the challenging and large scale M5 benchmark, a real-world retail dataset, TSMixer demonstrates superior performance compared to the state-of-the-art alternatives. Our results underline the importance of efficiently utilizing cross-variate and auxiliary information for improving the performance of time series forecasting. We present various analyses to shed light into the capabilities of TSMixer. The design paradigms utilized in TSMixer are expected to open new horizons for deep learning-based time series forecasting.

## In ModernTSF
Default config: `configs/models/TSMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{chen2023tsmixer,
  author        = {Si-An Chen and
                  Chun-Liang Li and
                  Nate Yoder and
                  Sercan O. Arik and
                  Tomas Pfister},
  title         = {TSMixer: An All-MLP Architecture for Time Series Forecasting},
  year          = {2023},
  eprint        = {2303.06053},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2303.06053}
}
```
