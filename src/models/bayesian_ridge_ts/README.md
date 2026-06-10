---
model: "BayesianRidgeTS"
forecasting_setting: "time_series"
config: "configs/models/BayesianRidgeTS.toml"
registry: "models.bayesian_ridge_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# BayesianRidgeTS

BayesianRidgeTS is a time series forecasting model for univariate and multivariate sequence prediction. It is a PyTorch-native linear predictor inspired by Bayesian ridge regression, applying stronger shrinkage regularisation over the input window to produce forecasts for the prediction horizon.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Bayesian ridge regression is a classical statistical technique that places a Gaussian prior over the regression weights, equivalent to L2 (ridge) regularisation with a prior variance determined by empirical Bayes or cross-validation. In the time-series setting each output channel is predicted independently by a linear map from the flattened input window; the Bayesian prior encourages compact, well-regularised weight matrices that generalise better under limited data. The ModernTSF implementation trains this model end-to-end as a `torch.nn.Module`, enabling use on GPU/MPS via the standard training loop and making it a strong classical baseline for comparison against deep forecasters.

## In ModernTSF
Default config: `configs/models/BayesianRidgeTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{mackay1992bayesian,
  author  = {David J. C. MacKay},
  title   = {Bayesian Interpolation},
  journal = {Neural Computation},
  volume  = {4},
  number  = {3},
  pages   = {415--447},
  year    = {1992},
  doi     = {10.1162/neco.1992.4.3.415},
  url     = {https://doi.org/10.1162/neco.1992.4.3.415}
}
```
