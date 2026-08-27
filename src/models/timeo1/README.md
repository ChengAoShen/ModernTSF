---
name: "TimeO1"
implementation: rewrite
summary: "TimeO1 is a time series forecasting approach that improves training through a transformation-augmented learning objective: it transforms the label sequence into decorrelated components ranked by significance, then trains the model to align only the most important components, addressing both label autocorrelation bias and the excessive task complexity that grows with the forecast horizon under standard mean squared error training."
paper:
  title: "Time-o1: Time-Series Forecasting Needs Transformed Label Alignment"
  venue: "NeurIPS 2025"
  year: 2025
  url: "https://arxiv.org/abs/2505.17847"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# TimeO1

TimeO1 is a time series forecasting approach that improves training through a transformation-augmented learning objective: it transforms the label sequence into decorrelated components ranked by significance, then trains the model to align only the most important components, addressing both label autocorrelation bias and the excessive task complexity that grows with the forecast horizon under standard mean squared error training.

<!-- model-card:canonical:start -->
## Method overview

TimeO1 is a time series forecasting approach that improves training through a transformation-augmented learning objective: it transforms the label sequence into decorrelated components ranked by significance, then trains the model to align only the most important components, addressing both label autocorrelation bias and the excessive task complexity that grows with the forecast horizon under standard mean squared error training.

## Core architecture

TimeO1 is a time series forecasting approach that improves training through a transformation-augmented learning objective: it transforms the label sequence into decorrelated components ranked by significance, then trains the model to align only the most important components, addressing both label autocorrelation bias and the excessive task complexity that grows with the forecast horizon under standard mean squared error training.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2505.17847); title: Time-o1: Time-Series Forecasting Needs Transformed Label Alignment; venue/year: NeurIPS 2025 / 2025
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TimeO1.toml`](../../../configs/models/TimeO1.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Time-o1: Time-Series Forecasting Needs Transformed Label Alignment
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.17847

## Abstract
Training time-series forecast models presents unique challenges in designing effective learning objectives. Existing methods predominantly utilize the temporal mean squared error, which faces two critical challenges: (1) label autocorrelation, which leads to bias from the label sequence likelihood; (2) excessive amount of tasks, which increases with the forecast horizon and complicates optimization. To address these challenges, we propose Time-o1, a transformation-augmented learning objective tailored for time-series forecasting. The central idea is to transform the label sequence into decorrelated components with discriminated significance. Models are then trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Extensive experiments demonstrate that Time-o1 achieves state-of-the-art performance and is compatible with various forecast models. Code is available at https://github.com/Master-PLC/Time-o1.

## In ModernTSF
Default config: `configs/models/TimeO1.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@misc{wang2025timeo,
  author        = {Hao Wang and
                  Licheng Pan and
                  Zhichao Chen and
                  Xu Chen and
                  Qingyang Dai and
                  Lei Wang and
                  Haoxuan Li and
                  Zhouchen Lin},
  title         = {Time-o1: Time-Series Forecasting Needs Transformed Label Alignment},
  year          = {2025},
  eprint        = {2505.17847},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.17847}
}
```
