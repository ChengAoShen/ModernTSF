---
model: "SVTime"
forecasting_setting: "time_series"
config: "configs/models/SVTime.toml"
registry: "models.svtime.registry"
paper_title: "SVTime: Small Time Series Forecasting Models Informed by \"Physics\" of Large Vision Model Forecasters"
venue: "arXiv preprint"
year: 2025
arxiv: "https://arxiv.org/abs/2510.09780"
---
# SVTime

SVTime is a compact, resource-efficient time-series forecasting model that distils key inductive biases — inter-period consistency, patch-wise variety, and distance-attenuating local attention — from large vision model (LVM) forecasters into small, meticulously crafted linear layers and constraint functions. It targets the long-term univariate/multivariate forecasting setting and achieves large-model-level accuracy with roughly 1000x fewer parameters than LVMs, making it suitable for resource-constrained environments.

## Paper
- **Title**: SVTime: Small Time Series Forecasting Models Informed by "Physics" of Large Vision Model Forecasters
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-10)
- **arXiv**: https://arxiv.org/abs/2510.09780

## Abstract
Time series AI is crucial for analyzing dynamic web content, driving a surge of pre-trained large models known for their strong knowledge encoding and transfer capabilities across diverse tasks. However, given their energy-intensive training, inference, and hardware demands, using large models as a one-fits-all solution raises serious concerns about carbon footprint and sustainability. For a specific task, a compact yet specialized, high-performing model may be more practical and affordable, especially for resource-constrained users such as small businesses. This motivates the question: Can we build cost-effective lightweight models with large-model-like performance on core tasks such as forecasting? This paper addresses this question by introducing SVTime, a novel Small model inspired by large Vision model (LVM) forecasters for long-term Time series forecasting (LTSF). Recently, LVMs have been shown as powerful tools for LTSF. We identify a set of key inductive biases of LVM forecasters -- analogous to the "physics" governing their behaviors in LTSF -- and design small models that encode these biases through meticulously crafted linear layers and constraint functions. Across 21 baselines spanning lightweight, complex, and pre-trained large models on 8 benchmark datasets, SVTime outperforms state-of-the-art (SOTA) lightweight models and rivals large models with 10^3 fewer parameters than LVMs, while enabling efficient training and inference in low-resource settings.

## In ModernTSF
Default config: `configs/models/SVTime.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{shen2025svtime,
  author        = {ChengAo Shen and
                  Ziming Zhao and
                  Hanghang Tong and
                  Dongjin Song and
                  Dongsheng Luo and
                  Qingsong Wen and
                  Jingchao Ni},
  title         = {SVTime: Small Time Series Forecasting Models Informed by "Physics" of Large Vision Model Forecasters},
  year          = {2025},
  eprint        = {2510.09780},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.09780}
}
```
