---
model: "GRUForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/GRUForecasterTS.toml"
registry: "models.gru_forecaster_ts.registry"
paper_title: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"
venue: "arXiv preprint"
year: 2014
arxiv: "https://arxiv.org/abs/1412.3555"
---
# GRUForecasterTS

GRUForecasterTS is a standard Gated Recurrent Unit (GRU) sequence-to-sequence forecaster registered for the time-series forecasting setting. It accepts a fixed-length historical window of univariate or multivariate values and produces a fixed-length forecast horizon by unrolling the GRU recurrence over the input and decoding the final hidden state.

## Paper
- **Title**: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **Venue**: arXiv preprint
- **Published**: 2014
- **arXiv**: https://arxiv.org/abs/1412.3555

## Abstract
In this paper we compare different types of recurrent units in recurrent neural networks (RNNs). Especially, we focus on more sophisticated units that implement a gating mechanism, such as a long short-term memory (LSTM) unit and a recently proposed gated recurrent unit (GRU). We evaluate these recurrent units on the tasks of polyphonic music modeling and speech signal modeling. Our experiments revealed that these advanced recurrent units are indeed better than more traditional recurrent units such as tanh units. Also, we found GRU to be comparable to LSTM.

## In ModernTSF
Default config: `configs/models/GRUForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
