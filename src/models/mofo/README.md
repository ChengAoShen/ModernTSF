---
model: "MoFo"
forecasting_setting: "time_series"
config: "configs/models/MoFo.toml"
registry: "models.mofo.registry"
paper_title: "MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling"
venue: "NeurIPS 2025"
year: 2025
arxiv: ""
---
# MoFo

MoFo is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It explicitly models periodic patterns by constructing period-structured 2D patch tensors through discrete sampling and introduces a period-aware modulator that applies a learnable regulated relaxation function to guide attention coefficients toward periodic trends, achieving high memory efficiency and fast training speed.

## Paper
- **Title**: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
The stable periodic patterns present in the time series data serve as the foundation for long-term forecasting. However, existing models suffer from limitations such as continuous and chaotic input partitioning, as well as weak inductive biases, which restrict their ability to capture such recurring structures. In this paper, we propose MoFo, which interprets periodicity as both the correlation of period-aligned time steps and the trend of period-offset time steps. We first design period-structured patches—2D tensors generated through discrete sampling—where each row contains only period-aligned time steps, enabling direct modeling of periodic correlations. Period-offset time steps within a period are aligned in columns. To capture trends across these offset time steps, we introduce a period-aware modulator. This modulator introduces an adaptive strong inductive bias through a regulated relaxation function, encouraging the model to generate attention coefficients that align with periodic trends. This function is end-to-end trainable, enabling the model to adaptively capture the distinct periodic patterns across diverse datasets. Extensive empirical results on widely used benchmark datasets demonstrate that MoFo achieves competitive performance while maintaining high memory efficiency and fast training speed.

## In ModernTSF
Default config: `configs/models/MoFo.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{ma2025mofo,
  author    = {Jiaming Ma and Binwu Wang and Qihe Huang and Guanjun Wang and Pengkun Wang and Zhengyang Zhou and Yang Wang},
  title     = {{MoFo}: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MoFo}
}
```
