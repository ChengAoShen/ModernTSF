---
name: "ETSformer"
implementation: upstream
summary: "ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting. It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality."
paper:
  title: "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting"
  venue: "arXiv preprint"
  year: 2022
  url: "https://arxiv.org/abs/2202.01381"
codebase:
  url: "https://github.com/thuml/Time-Series-Library"
  revision: "230805fe9f451b61e34b96116d995b417e343ac0"
  license: "MIT"
  usage: ported
---
# ETSformer

ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting. It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality.

<!-- model-card:canonical:start -->
## Method overview

ETSformer is a time series forecasting model that combines classical exponential smoothing principles with the Transformer architecture to address limitations of vanilla Transformers for long-term forecasting.

## Core architecture

It introduces two novel attention mechanisms—exponential smoothing attention (ESA) and frequency attention (FA)—to replace standard self-attention, and redesigns the Transformer with modular decomposition blocks that learn to separate time series into interpretable components: level, growth, and seasonality.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2202.01381); title: ETSformer: Exponential Smoothing Transformers for Time-series Forecasting; venue/year: arXiv preprint / 2022
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `230805fe9f451b61e34b96116d995b417e343ac0`; license: `MIT`; usage: `ported`

## Local implementation

This card declares a `upstream` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/ETSformer.toml`](../../../configs/models/ETSformer.toml).

## Differences

Implementation: **upstream** (numerical parity pending). The implementation is pinned to
[`thuml/Time-Series-Library`](https://github.com/thuml/Time-Series-Library)
revision `230805fe9f451b61e34b96116d995b417e343ac0` under MIT and corresponds to
the authors' ETSformer release. Exponential-smoothing attention, growth and
Fourier-seasonality decomposition, damping, level updates, and decoder
aggregation are retained. Only long-term forecasting is included. Output width
is fixed to `enc_in` because the upstream level residual requires that equality;
the previously accepted incompatible `c_out` override was removed. Dead
feed-forward and normalization parameters in the terminal encoder layer are
also omitted because that layer's residual state is never consumed.

## Shared components

- [`embed`](../../components/embed.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_layers=2`, `d_ff=256`, `top_k=3`, `dropout=0.1`, `activation='sigmoid'`, `embed='timeF'`, `freq='h'`
<!-- model-card:canonical:end -->

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

Implementation: **upstream** (numerical parity pending). The implementation is pinned to
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
