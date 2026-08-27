---
name: "TimeXer"
implementation: rewrite
summary: "TimeXer is a Transformer-based time series forecasting model for the standard time series forecasting setting that extends canonical Transformers to handle exogenous variables. It introduces deftly designed embedding layers that separately represent endogenous (target) variables via patch-wise self-attention and exogenous (external) variables via variate-wise cross-attention, with learned global endogenous tokens bridging causal information from exogenous series into endogenous temporal patches."
paper:
  title: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
  venue: "NeurIPS 2024"
  year: 2024
  url: "https://arxiv.org/abs/2402.19072"
codebase:
  url: "https://github.com/thuml/TimeXer"
  revision: "76011909357972bd55a27adba2e1be994d81b327"
  license: "NOASSERTION"
  usage: reference-only
---
# TimeXer

TimeXer is a Transformer-based time series forecasting model for the standard time series forecasting setting that extends canonical Transformers to handle exogenous variables. It introduces deftly designed embedding layers that separately represent endogenous (target) variables via patch-wise self-attention and exogenous (external) variables via variate-wise cross-attention, with learned global endogenous tokens bridging causal information from exogenous series into endogenous temporal patches.

<!-- model-card:canonical:start -->
## Method overview

TimeXer is a Transformer-based time series forecasting model for the standard time series forecasting setting that extends canonical Transformers to handle exogenous variables.

## Core architecture

It introduces deftly designed embedding layers that separately represent endogenous (target) variables via patch-wise self-attention and exogenous (external) variables via variate-wise cross-attention, with learned global endogenous tokens bridging causal information from exogenous series into endogenous temporal patches.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2402.19072); title: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables; venue/year: NeurIPS 2024 / 2024
- [codebase](https://github.com/thuml/TimeXer); revision: `76011909357972bd55a27adba2e1be994d81b327`; license: `NOASSERTION`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TimeXer.toml`](../../../configs/models/TimeXer.toml).

## Differences

Compared against the author repository at commit `76011909357972bd55a27adba2e1be994d81b327`. Endogenous patch/global-token attention and cross-attention to exogenous variables are retained, but ModernTSF infers channel roles from feature mode and ordering. The repository has no explicit code license and no numerical parity evidence, so this model remains pending implementation audit.

## Shared components

- [`embed`](../../components/embed.py)
- [`flatten_forecast_head`](../../components/flatten_forecast_head.py)
- [`self_attention_family`](../../components/self_attention_family.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `patch_len=16`, `dropout=0.1`, `factor=3`, `activation='gelu'`, `use_norm=True`, `embed='timeF'`, `freq='h'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.19072

## Abstract
Deep models have demonstrated remarkable performance in time series forecasting. However, due to the partially-observed nature of real-world applications, solely focusing on the target of interest, so-called endogenous variables, is usually insufficient to guarantee accurate forecasting. Notably, a system is often recorded into multiple variables, where the exogenous variables can provide valuable external information for endogenous variables. Thus, unlike well-established multivariate or univariate forecasting paradigms that either treat all the variables equally or ignore exogenous information, this paper focuses on a more practical setting: time series forecasting with exogenous variables. We propose a novel approach, TimeXer, to ingest external information to enhance the forecasting of endogenous variables. With deftly designed embedding layers, TimeXer empowers the canonical Transformer with the ability to reconcile endogenous and exogenous information, where patch-wise self-attention and variate-wise cross-attention are used simultaneously. Moreover, global endogenous tokens are learned to effectively bridge the causal information underlying exogenous series into endogenous temporal patches. Experimentally, TimeXer achieves consistent state-of-the-art performance on twelve real-world forecasting benchmarks and exhibits notable generality and scalability. Code is available at this repository: https://github.com/thuml/TimeXer.

## In ModernTSF
Default config: `configs/models/TimeXer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

Compared against the author repository at commit `76011909357972bd55a27adba2e1be994d81b327`. Endogenous patch/global-token attention and cross-attention to exogenous variables are retained, but ModernTSF infers channel roles from feature mode and ordering. The repository has no explicit code license and no numerical parity evidence, so this model remains pending implementation audit.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/WangWDQZLQWL24,
  author       = {Yuxuan Wang and
                  Haixu Wu and
                  Jiaxiang Dong and
                  Guo Qin and
                  Haoran Zhang and
                  Yong Liu and
                  Yunzhong Qiu and
                  Jianmin Wang and
                  Mingsheng Long},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {TimeXer: Empowering Transformers for Time Series Forecasting with
                  Exogenous Variables},
  booktitle    = {Advances in Neural Information Processing Systems 37: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html},
  timestamp    = {Tue, 26 May 2026 17:12:08 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/WangWDQZLQWL24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
