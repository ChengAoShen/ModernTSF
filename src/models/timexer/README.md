---
model: "TimeXer"
forecasting_setting: "time_series"
config: "configs/models/TimeXer.toml"
registry: "models.timexer.registry"
paper_title: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2402.19072"
---
# TimeXer

TimeXer is a Transformer-based time series forecasting model for the standard time series forecasting setting that extends canonical Transformers to handle exogenous variables. It introduces deftly designed embedding layers that separately represent endogenous (target) variables via patch-wise self-attention and exogenous (external) variables via variate-wise cross-attention, with learned global endogenous tokens bridging causal information from exogenous series into endogenous temporal patches.

## Paper
- **Title**: TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.19072

## Abstract
Deep models have demonstrated remarkable performance in time series forecasting. However, due to the partially-observed nature of real-world applications, solely focusing on the target of interest, so-called endogenous variables, is usually insufficient to guarantee accurate forecasting. Notably, a system is often recorded into multiple variables, where the exogenous variables can provide valuable external information for endogenous variables. Thus, unlike well-established multivariate or univariate forecasting paradigms that either treat all the variables equally or ignore exogenous information, this paper focuses on a more practical setting: time series forecasting with exogenous variables. We propose a novel approach, TimeXer, to ingest external information to enhance the forecasting of endogenous variables. With deftly designed embedding layers, TimeXer empowers the canonical Transformer with the ability to reconcile endogenous and exogenous information, where patch-wise self-attention and variate-wise cross-attention are used simultaneously. Moreover, global endogenous tokens are learned to effectively bridge the causal information underlying exogenous series into endogenous temporal patches. Experimentally, TimeXer achieves consistent state-of-the-art performance on twelve real-world forecasting benchmarks and exhibits notable generality and scalability. Code is available at this repository: https://github.com/thuml/TimeXer.

## In ModernTSF
Default config: `configs/models/TimeXer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
