---
model: "RidgeRegressionTS"
forecasting_setting: "time_series"
config: "configs/models/RidgeRegressionTS.toml"
registry: "models.ridge_regression_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# RidgeRegressionTS

RidgeRegressionTS is a PyTorch-native adapter that implements ridge regression (L2-regularized linear regression) as a time series forecasting model, mapping a lagged feature window to the prediction horizon through a learned linear projection with L2 weight penalty, running through the standard ModernTSF trainer and supporting GPU acceleration.

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Ridge regression is a classical regularized linear model that extends ordinary least squares by adding an L2 penalty on the regression coefficients (Tikhonov regularization). Applied to time series forecasting, the model treats lagged values of all channels as input features and predicts the future horizon via a single linear layer whose weights are regularized to avoid overfitting. The L2 penalty shrinks large coefficients toward zero, improving generalization on high-dimensional or correlated feature sets. In the ModernTSF context, the model is implemented as a `torch.nn.Module` with a learnable linear layer and a configurable regularization strength, enabling GPU-accelerated training through the standard benchmark trainer alongside all other model classes.

## In ModernTSF
Default config: `configs/models/RidgeRegressionTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{hoerl1970ridge,
  author  = {Arthur E. Hoerl and Robert W. Kennard},
  title   = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  volume  = {12},
  number  = {1},
  pages   = {55--67},
  year    = {1970},
  doi     = {10.1080/00401706.1970.10488634}
}
```
