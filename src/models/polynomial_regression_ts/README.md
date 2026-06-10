---
model: "PolynomialRegressionTS"
forecasting_setting: "time_series"
config: "configs/models/PolynomialRegressionTS.toml"
registry: "models.polynomial_regression_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# PolynomialRegressionTS

PolynomialRegressionTS is a time series forecasting model for univariate and multivariate sequence prediction. It extends linear regression by constructing polynomial lag features — raw, squared, and square-root transformations of the input window — and learning a linear map from these features to the forecast horizon.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Polynomial regression is a classical statistical technique that enriches the feature space of a linear model by including nonlinear transformations of the input variables. In the time-series forecasting context, the historical window values are expanded with squared and square-root lag features before a linear predictor maps them to the output horizon. This polynomial feature augmentation allows the model to capture simple nonlinear trends without the overhead of a deep neural network. The ModernTSF implementation trains this model end-to-end as a `torch.nn.Module`, enabling execution on GPU/MPS via the standard training loop and making it a useful nonlinear classical baseline alongside purely linear methods.

## In ModernTSF
Default config: `configs/models/PolynomialRegressionTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@book{draper1998applied,
  author    = {Norman R. Draper and Harry Smith},
  title     = {Applied Regression Analysis},
  edition   = {3rd},
  publisher = {Wiley},
  address   = {New York},
  year      = {1998},
  doi       = {10.1002/9781118625590}
}
```
