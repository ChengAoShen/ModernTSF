---
model: "DynamicTMoE"
forecasting_setting: "time_series"
config: "configs/models/DynamicTMoE.toml"
registry: "models.dynamic_tmoe.registry"
paper_title: "Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting"
venue: "ICML 2026"
year: 2026
arxiv: ""
---
# DynamicTMoE

DynamicTMoE is a drift-aware dynamic Mixture-of-Experts framework for non-stationary multivariate time series forecasting in the standard time-series setting. It overcomes the rigidity of traditional MoE architectures by using Maximum Mean Discrepancy (MMD) to detect distribution shifts, and dynamically expanding or pruning a heterogeneous expert pool at runtime — allowing the model to continuously adapt its capacity to changing data distributions. ModernTSF registers a lightweight native adapter that follows the shared prediction interface and normalization path from `src/models/_recent_tsf.py`.

## Paper
- **Title**: Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting
- **Venue**: ICML 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
Dynamic TMoE introduces an adaptive Mixture of Experts framework designed for time series forecasting in non-stationary environments. The method uses Maximum Mean Discrepancy (MMD) to detect distribution shifts and responds by dynamically expanding or pruning a heterogeneous expert pool, overcoming the rigidity of traditional fixed-capacity MoE designs. A drift-aware routing mechanism selects or allocates experts based on detected statistical changes in the input distribution, enabling robust forecasting under concept drift. The framework was accepted as a poster at the Forty-third International Conference on Machine Learning (ICML 2026) and demonstrates notable improvements in MSE and MAE across nine standard benchmarks compared to prior state-of-the-art methods. The official implementation is available at https://github.com/andone-07/Dynamic-TMoE.

## In ModernTSF
Default config: `configs/models/DynamicTMoE.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{zhu2026dynamictmoe,
  author        = {Jiawen Zhu and Shuhan Liu and Di Weng and Yingcai Wu},
  title         = {Dynamic TMoE: {A} Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting},
  year          = {2026},
  eprint        = {2605.20678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.20678}
}
```
