---
model: "NHiTS"
forecasting_setting: "time_series"
config: "configs/models/NHiTS.toml"
registry: "models.nhits.registry"
paper_title: "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"
venue: "AAAI 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2201.12886"
---
# NHiTS

NHiTS (Neural Hierarchical Interpolation for Time Series) is a time-series forecasting model that addresses long-horizon prediction by stacking MLP blocks with multi-rate data sampling and hierarchical interpolation. Each block in the stack emphasises a different frequency band of the signal, and the blocks' outputs are combined to synthesise the final forecast.

## Paper
- **Title**: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-01)
- **arXiv**: https://arxiv.org/abs/2201.12886

## Abstract
Recent progress in neural forecasting accelerated improvements in the performance of large-scale forecasting systems. Yet, long-horizon forecasting remains a very difficult task. Two common challenges afflicting the task are the volatility of the predictions and their computational complexity. We introduce N-HiTS, a model which addresses both challenges by incorporating novel hierarchical interpolation and multi-rate data sampling techniques. These techniques enable the proposed method to assemble its predictions sequentially, emphasizing components with different frequencies and scales while decomposing the input signal and synthesizing the forecast. We prove that the hierarchical interpolation technique can efficiently approximate arbitrarily long horizons in the presence of smoothness. Additionally, we conduct extensive large-scale dataset experiments from the long-horizon forecasting literature, demonstrating the advantages of our method over the state-of-the-art methods, where N-HiTS provides an average accuracy improvement of almost 20% over the latest Transformer architectures while reducing the computation time by an order of magnitude (50 times).

## In ModernTSF
Default config: `configs/models/NHiTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{challu2022nhits,
  author        = {Cristian Challu and
                  Kin G. Olivares and
                  Boris N. Oreshkin and
                  Federico Garza and
                  Max Mergenthaler-Canseco and
                  Artur Dubrawski},
  title         = {N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting},
  year          = {2022},
  eprint        = {2201.12886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2201.12886}
}
```
