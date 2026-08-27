---
name: "GRUForecasterTS"
implementation: rewrite
summary: "GRUForecasterTS is a standard Gated Recurrent Unit (GRU) sequence-to-sequence forecaster registered for the time-series forecasting setting. It accepts a fixed-length historical window of univariate or multivariate values and produces a fixed-length forecast horizon by unrolling the GRU recurrence over the input and decoding the final hidden state."
paper:
  title: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
  venue: "arXiv preprint"
  year: 2014
  url: "https://arxiv.org/abs/1412.3555"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# GRUForecasterTS

GRUForecasterTS is a standard Gated Recurrent Unit (GRU) sequence-to-sequence forecaster registered for the time-series forecasting setting. It accepts a fixed-length historical window of univariate or multivariate values and produces a fixed-length forecast horizon by unrolling the GRU recurrence over the input and decoding the final hidden state.

<!-- model-card:canonical:start -->
## Method overview

GRUForecasterTS is a standard Gated Recurrent Unit (GRU) sequence-to-sequence forecaster registered for the time-series forecasting setting.

## Core architecture

It accepts a fixed-length historical window of univariate or multivariate values and produces a fixed-length forecast horizon by unrolling the GRU recurrence over the input and decoding the final hidden state.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1412.3555); title: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling; venue/year: arXiv preprint / 2014
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/GRUForecasterTS.toml`](../../../configs/models/GRUForecasterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.0`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **Venue**: arXiv preprint
- **Published**: 2014
- **arXiv**: https://arxiv.org/abs/1412.3555

## Abstract
In this paper we compare different types of recurrent units in recurrent neural networks (RNNs). Especially, we focus on more sophisticated units that implement a gating mechanism, such as a long short-term memory (LSTM) unit and a recently proposed gated recurrent unit (GRU). We evaluate these recurrent units on the tasks of polyphonic music modeling and speech signal modeling. Our experiments revealed that these advanced recurrent units are indeed better than more traditional recurrent units such as tanh units. Also, we found GRU to be comparable to LSTM.

## In ModernTSF
Default config: `configs/models/GRUForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@misc{chung2014empirical,
  author        = {Junyoung Chung and
                  Caglar Gulcehre and
                  KyungHyun Cho and
                  Yoshua Bengio},
  title         = {Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling},
  year          = {2014},
  eprint        = {1412.3555},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1412.3555}
}
```
