---
name: "MoFo"
implementation: upstream
summary: "MoFo is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It explicitly models periodic patterns by constructing period-structured 2D patch tensors through discrete sampling and introduces a period-aware modulator that applies a learnable regulated relaxation function to guide attention coefficients toward periodic trends, achieving high memory efficiency and fast training speed."
paper:
  title: "MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling"
  venue: "NeurIPS 2025"
  year: 2025
  url: "https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html"
codebase:
  url: "https://github.com/PoorOtterBob/MoFo"
  revision: "2d14b47ea839c3809952b412340d72393f2521dc"
  license: "MIT"
  usage: ported
---
# MoFo

MoFo is a Transformer-based long-term time-series forecasting model for the standard time-series setting. It explicitly models periodic patterns by constructing period-structured 2D patch tensors through discrete sampling and introduces a period-aware modulator that applies a learnable regulated relaxation function to guide attention coefficients toward periodic trends, achieving high memory efficiency and fast training speed.

<!-- model-card:canonical:start -->
## Method overview

MoFo is a Transformer-based long-term time-series forecasting model for the standard time-series setting.

## Core architecture

It explicitly models periodic patterns by constructing period-structured 2D patch tensors through discrete sampling and introduces a period-aware modulator that applies a learnable regulated relaxation function to guide attention coefficients toward periodic trends, achieving high memory efficiency and fast training speed.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html); title: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling; venue/year: NeurIPS 2025 / 2025
- [codebase](https://github.com/PoorOtterBob/MoFo); revision: `2d14b47ea839c3809952b412340d72393f2521dc`; license: `MIT`; usage: `ported`

## Local implementation

This card declares a `upstream` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/MoFo.toml`](../../../configs/models/MoFo.toml).

## Differences

This **upstream port** uses the official MIT-licensed implementation at revision
`2d14b47ea839c3809952b412340d72393f2521dc`. The local core keeps the forecast
path and removes unrelated task branches; the adapter converts raw calendar
marks to the TFB normalization expected upstream. Supported periods are 24,
96, 144, and 288. Dataset preparation, loss, optimizer, and experiment defaults
belong to the shared benchmark runner rather than the upstream scripts.

Pinned-source parity for the default period-24 forecast path is recorded in
`verification/parity/MoFo.json`, including eval/train outputs, defining
intermediates, input and active-parameter gradients, batch sizes 1/2, wrapper
preprocessing, and state-dict round trips. Removed task-branch parameters are
listed as upstream-only inactive state in the detailed artifact.

## Shared components

- [`marks`](../../components/marks.py)
- [`revin`](../../components/revin.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=6`, `d_model=64`, `periodic=24`, `head=4`, `d_layers=1`, `bias=1`, `cias=1`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
The stable periodic patterns present in the time series data serve as the foundation for long-term forecasting. However, existing models suffer from limitations such as continuous and chaotic input partitioning, as well as weak inductive biases, which restrict their ability to capture such recurring structures. In this paper, we propose MoFo, which interprets periodicity as both the correlation of period-aligned time steps and the trend of period-offset time steps. We first design period-structured patches—2D tensors generated through discrete sampling—where each row contains only period-aligned time steps, enabling direct modeling of periodic correlations. Period-offset time steps within a period are aligned in columns. To capture trends across these offset time steps, we introduce a period-aware modulator. This modulator introduces an adaptive strong inductive bias through a regulated relaxation function, encouraging the model to generate attention coefficients that align with periodic trends. This function is end-to-end trainable, enabling the model to adaptively capture the distinct periodic patterns across diverse datasets. Extensive empirical results on widely used benchmark datasets demonstrate that MoFo achieves competitive performance while maintaining high memory efficiency and fast training speed.

## In ModernTSF
Default config: `configs/models/MoFo.toml`; model specification: `spec.py`; local runtime implementation: `model.py`.

## Verification

This **upstream port** uses the official MIT-licensed implementation at revision
`2d14b47ea839c3809952b412340d72393f2521dc`. The local core keeps the forecast
path and removes unrelated task branches; the adapter converts raw calendar
marks to the TFB normalization expected upstream. Supported periods are 24,
96, 144, and 288. Dataset preparation, loss, optimizer, and experiment defaults
belong to the shared benchmark runner rather than the upstream scripts.

Pinned-source parity for the default period-24 forecast path is recorded in
`verification/parity/MoFo.json`, including eval/train outputs, defining
intermediates, input and active-parameter gradients, batch sizes 1/2, wrapper
preprocessing, and state-dict round trips. Removed task-branch parameters are
listed as upstream-only inactive state in the detailed artifact.

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
