---
model: "SEMPO"
forecasting_setting: "time_series"
config: "configs/models/SEMPO.toml"
registry: "models.sempo.registry"
paper_title: "SEMPO: Lightweight Foundation Models for Time Series Forecasting"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2510.19710"
---
# SEMPO

SEMPO is a lightweight time-series foundation model accepted at NeurIPS 2025. It combines an energy-aware spectral decomposition module that captures both high- and low-energy frequency signals with a Mixture-of-Prompts enabled Transformer that routes tokens to small dataset-specific prompt-based experts, enabling strong zero-shot and few-shot generalization across diverse datasets while requiring far less pre-training data and a smaller model size than existing foundation models.

## Paper
- **Title**: SEMPO: Lightweight Foundation Models for Time Series Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.19710

## Abstract
The recent boom of large pre-trained models witnesses remarkable success in developing foundation models (FMs) for time series forecasting. Despite impressive performance across diverse downstream forecasting tasks, existing time series FMs possess massive network architectures and require substantial pre-training on large-scale datasets, which significantly hinders their deployment in resource-constrained environments. In response to this growing tension between versatility and affordability, we propose SEMPO, a novel lightweight foundation model that requires pretraining on relatively small-scale data, yet exhibits strong general time series forecasting. Concretely, SEMPO comprises two key modules: 1) energy-aware SpEctral decomposition module, that substantially improves the utilization of pre-training data by modeling not only the high-energy frequency signals but also the low-energy yet informative frequency signals that are ignored in current methods; and 2) Mixture-of-PrOmpts enabled Transformer, that learns heterogeneous temporal patterns through small dataset-specific prompts and adaptively routes time series tokens to prompt-based experts for parameter-efficient model adaptation across different datasets and domains. Equipped with these modules, SEMPO significantly reduces both pre-training data scale and model size, while achieving strong generalization. Extensive experiments on two large-scale benchmarks covering 16 datasets demonstrate the superior performance of SEMPO in both zero-shot and few-shot forecasting scenarios compared with state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/SEMPO.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{he2025sempo,
  author        = {Hui He and
                  Kun Yi and
                  Yuanchi Ma and
                  Qi Zhang and
                  Zhendong Niu and
                  Guansong Pang},
  title         = {SEMPO: Lightweight Foundation Models for Time Series Forecasting},
  year          = {2025},
  eprint        = {2510.19710},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.19710}
}
```
