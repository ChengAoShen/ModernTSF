---
model: "DeepAR"
forecasting_setting: "time_series"
config: "configs/models/DeepAR.toml"
registry: "models.deepar.registry"
paper_title: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
venue: "International Journal of Forecasting 2020"
year: 2020
arxiv: "https://arxiv.org/abs/1704.04110"
---
# DeepAR

DeepAR is an autoregressive recurrent neural network designed for probabilistic time-series forecasting. It trains a single global LSTM-based model over many related time series and outputs a learned probability distribution over the forecast horizon rather than a point prediction, making it well-suited to the standard univariate and multivariate time-series forecasting setting.

## Paper
- **Title**: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
- **Venue**: International Journal of Forecasting 2020
- **Published**: 2020 (arXiv: 2017-04)
- **arXiv**: https://arxiv.org/abs/1704.04110

## Abstract
Probabilistic forecasting, i.e. estimating the probability distribution of a time series' future given its past, is a key enabler for optimizing business processes. In retail businesses, for example, forecasting demand is crucial for having the right inventory available at the right time at the right place. In this paper we propose DeepAR, a methodology for producing accurate probabilistic forecasts, based on training an auto regressive recurrent network model on a large number of related time series. We demonstrate how by applying deep learning techniques to forecasting, one can overcome many of the challenges faced by widely-used classical approaches to the problem. We show through extensive empirical evaluation on several real-world forecasting data sets accuracy improvements of around 15% compared to state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/DeepAR.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/FlunkertSG17,
  author       = {Valentin Flunkert and
                  David Salinas and
                  Jan Gasthaus},
  title        = {DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks},
  journal      = {CoRR},
  volume       = {abs/1704.04110},
  year         = {2017},
  url          = {http://arxiv.org/abs/1704.04110},
  eprinttype   = {arXiv},
  eprint       = {1704.04110},
  timestamp    = {Mon, 13 Aug 2018 16:46:25 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/FlunkertSG17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
