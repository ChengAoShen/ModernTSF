---
model: "ETSformer"
forecasting_setting: "time_series"
config: "configs/models/ETSformer.toml"
spec: "models.etsformer.spec"
paper_title: "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting"
venue: "arXiv preprint"
year: 2022
arxiv: "https://arxiv.org/abs/2202.01381"
---
# ETSformer

ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting. It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality.

## Paper
- **Title**: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2022
- **arXiv**: https://arxiv.org/abs/2202.01381

## Abstract
Transformers have been actively studied for time-series forecasting in recent years. While often showing promising results in various scenarios, traditional Transformers are not designed to fully exploit the characteristics of time-series data and thus suffer some fundamental limitations, e.g., they generally lack of decomposition capability and interpretability, and are neither effective nor efficient for long-term forecasting. In this paper, we propose ETSFormer, a novel time-series Transformer architecture, which exploits the principle of exponential smoothing in improving Transformers for time-series forecasting. In particular, inspired by the classical exponential smoothing methods in time-series forecasting, we propose the novel exponential smoothing attention (ESA) and frequency attention (FA) to replace the self-attention mechanism in vanilla Transformers, thus improving both accuracy and efficiency. Based on these, we redesign the Transformer architecture with modular decomposition blocks such that it can learn to decompose the time-series data into interpretable time-series components such as level, growth and seasonality. Extensive experiments on various time-series benchmarks validate the efficacy and advantages of the proposed method. Code is available at this https URL.

## In ModernTSF
Default config: `configs/models/ETSformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Evidence level: **upstream-port**. The implementation is pinned to
[`thuml/Time-Series-Library`](https://github.com/thuml/Time-Series-Library)
revision `230805fe9f451b61e34b96116d995b417e343ac0` under MIT and corresponds to
the authors' ETSformer release. Exponential-smoothing attention, growth and
Fourier-seasonality decomposition, damping, level updates, and decoder
aggregation are retained. Only long-term forecasting is included. Output width
is fixed to `enc_in` because the upstream level residual requires that equality;
the previously accepted incompatible `c_out` override was removed. Dead
feed-forward and normalization parameters in the terminal encoder layer are
also omitted because that layer's residual state is never consumed.

## Citation

```bibtex
@misc{woo2022etsformer,
  author        = {Gerald Woo and
                  Chenghao Liu and
                  Doyen Sahoo and
                  Akshat Kumar and
                  Steven Hoi},
  title         = {ETSformer: Exponential Smoothing Transformers for Time-series Forecasting},
  year          = {2022},
  eprint        = {2202.01381},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2202.01381}
}
```
