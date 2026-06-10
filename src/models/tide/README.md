---
model: "TiDE"
forecasting_setting: "time_series"
config: "configs/models/TiDE.toml"
registry: "models.tide.registry"
paper_title: "Long-term Forecasting with TiDE: Time-series Dense Encoder"
venue: "TMLR 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2304.08424"
---
# TiDE

TiDE (Time-series Dense Encoder) is an MLP-based encoder-decoder model for long-term time series forecasting, serving the standard time series prediction setting with optional covariate support. It encodes the historical time series together with past and future covariates using dense MLP layers, then decodes to produce future predictions — combining the simplicity and speed of linear models with the expressiveness needed for nonlinear dependencies. TiDE is 5-10x faster than comparable Transformer-based models on standard benchmarks.

## Paper
- **Title**: Long-term Forecasting with TiDE: Time-series Dense Encoder
- **Venue**: TMLR 2023
- **Published**: 2023 (arXiv: 2023-04)
- **arXiv**: https://arxiv.org/abs/2304.08424

## Abstract
Recent work has shown that simple linear models can outperform several Transformer based approaches in long term time-series forecasting. Motivated by this, we propose a Multi-layer Perceptron (MLP) based encoder-decoder model, Time-series Dense Encoder (TiDE), for long-term time-series forecasting that enjoys the simplicity and speed of linear models while also being able to handle covariates and non-linear dependencies. Theoretically, we prove that the simplest linear analogue of our model can achieve near optimal error rate for linear dynamical systems (LDS) under some assumptions. Empirically, we show that our method can match or outperform prior approaches on popular long-term time-series forecasting benchmarks while being 5-10x faster than the best Transformer based model.

## In ModernTSF
Default config: `configs/models/TiDE.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/tmlr/DasKLMSY23,
  author       = {Abhimanyu Das and
                  Weihao Kong and
                  Andrew Leach and
                  Shaan Mathur and
                  Rajat Sen and
                  Rose Yu},
  title        = {Long-term Forecasting with TiDE: Time-series Dense Encoder},
  journal      = {Trans. Mach. Learn. Res.},
  volume       = {2023},
  year         = {2023},
  url          = {https://arxiv.org/abs/2304.08424},
  eprinttype   = {arXiv},
  eprint       = {2304.08424},
  timestamp    = {Thu, 01 Aug 2024 15:37:25 +0200},
  biburl       = {https://dblp.org/rec/journals/tmlr/DasKLMSY23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
