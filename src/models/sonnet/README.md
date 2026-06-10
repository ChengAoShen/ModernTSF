---
model: "Sonnet"
forecasting_setting: "time_series"
config: "configs/models/Sonnet.toml"
registry: "models.sonnet.registry"
paper_title: "Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2505.15312"
---
# Sonnet

Sonnet (Spectral Operator Neural Network) is a time series forecasting model for multivariate prediction. It applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. The core of its predictive skill is Multivariable Coherence Attention (MVCA), which leverages spectral coherence among variables to model inter-variable dependencies in the frequency domain, avoiding the pitfalls of naive self-attention for time series.

## Paper
- **Title**: Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting
- **Venue**: AAAI 2026 (Oral)
- **Published**: 2026 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.15312

## Abstract
Multivariable time series forecasting methods can integrate information from exogenous variables, leading to significant prediction accuracy gains. The transformer architecture has been widely applied in various time series forecasting models due to its ability to capture long-range sequential dependencies. However, a naïve application of transformers often struggles to effectively model complex relationships among variables over time. To mitigate against this, we propose a novel architecture, termed Spectral Operator Neural Network (Sonnet). Sonnet applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. Its predictive skill relies on the Multivariable Coherence Attention (MVCA), an operation that leverages spectral coherence to model variable dependencies. Our empirical analysis shows that Sonnet yields the best performance on 34 out of 47 forecasting tasks with an average mean absolute error (MAE) reduction of 2.2% against the most competitive baseline. We further show that MVCA can remedy the deficiencies of naïve attention in various deep learning models, reducing MAE by 10.7% on average in the most challenging forecasting tasks.

## In ModernTSF
Default config: `configs/models/Sonnet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ShuL26,
  author       = {Yuxuan Shu and
                  Vasileios Lampos},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Sonnet: Spectral Operator Neural Network for Multivariable Time Series
                  Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {25419--25427},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i30.39736},
  doi          = {10.1609/AAAI.V40I30.39736},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ShuL26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
