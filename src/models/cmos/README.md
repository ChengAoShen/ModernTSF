---
model: "CMoS"
forecasting_setting: "time_series"
config: "configs/models/CMoS.toml"
registry: "models.cmos.registry"
paper_title: "CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations"
venue: "arXiv preprint"
year: 2025
arxiv: "https://arxiv.org/abs/2505.19090"
---
# CMoS

CMoS is a super-lightweight multivariate time series forecasting model for the standard time-series setting. Rather than learning shape embeddings, it directly models spatial correlations between different time-series chunks using a Correlation Mixing strategy that captures diverse channel dependencies with minimal parameters, and an optional Periodicity Injection technique for faster convergence — achieving competitive accuracy at up to 100x the parameter efficiency of DLinear.

## Paper
- **Title**: CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.19090

## Abstract
Recent advances in lightweight time series forecasting models suggest the inherent simplicity of time series forecasting tasks. In this paper, we present CMoS, a super-lightweight time series forecasting model. Instead of learning the embedding of the shapes, CMoS directly models the spatial correlations between different time series chunks. Additionally, we introduce a Correlation Mixing technique that enables the model to capture diverse spatial correlations with minimal parameters, and an optional Periodicity Injection technique to ensure faster convergence. Despite utilizing as low as 1% of the lightweight model DLinear's parameters count, experimental results demonstrate that CMoS outperforms existing state-of-the-art models across multiple datasets. Furthermore, the learned weights of CMoS exhibit great interpretability, providing practitioners with valuable insights into temporal structures within specific application scenarios.

## In ModernTSF
Default config: `configs/models/CMoS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/SiPLPX25,
  author       = {Haotian Si and
                  Changhua Pei and
                  Jianhui Li and
                  Dan Pei and
                  Gaogang Xie},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise
                  Spatial Correlations},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/si25a.html},
  timestamp    = {Wed, 04 Feb 2026 17:22:46 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/SiPLPX25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
