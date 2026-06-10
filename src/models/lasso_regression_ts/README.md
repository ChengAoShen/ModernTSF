---
model: "LassoRegressionTS"
forecasting_setting: "time_series"
config: "configs/models/LassoRegressionTS.toml"
registry: "models.lasso_regression_ts.registry"
paper_title: "Regression Shrinkage and Selection via the Lasso"
venue: "Journal of the Royal Statistical Society: Series B, 1996"
year: 1996
arxiv: ""
---
# LassoRegressionTS

LassoRegressionTS is a PyTorch-native adapter that applies Lasso (L1-regularised linear) regression for time-series forecasting. It treats the look-back window as a flat lag feature vector and fits a linear projection to the prediction horizon, with L1 regularisation promoting sparsity over lag features. Running the linear layer as a `torch.nn.Module` allows training on CPU, CUDA, or MPS with the standard ModernTSF trainer.

## Paper
- **Title**: Regression Shrinkage and Selection via the Lasso
- **Venue**: Journal of the Royal Statistical Society: Series B, 1996
- **Published**: 1996
- **arXiv**: N/A

## Abstract
Lasso (Least Absolute Shrinkage and Selection Operator) is a classical penalised regression method introduced by Tibshirani (1996). It minimises the residual sum of squares subject to the sum of the absolute values of the regression coefficients being less than a constant. This L1 constraint has the effect of shrinking some coefficients exactly to zero, producing sparse and interpretable models while avoiding the instability of ordinary subset selection. The method combines the variable-selection capability of subset regression with the continuous shrinkage of ridge regression, making it effective when only a small subset of predictors is truly informative. In the time-series forecasting setting, Lasso regression is applied channel-by-channel over lag features derived from the historical input window, using L1 regularisation to identify the most predictive lags for each output channel.

## In ModernTSF
Default config: `configs/models/LassoRegressionTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{tibshirani1996regression,
  author  = {Robert Tibshirani},
  title   = {Regression Shrinkage and Selection via the Lasso},
  journal = {Journal of the Royal Statistical Society: Series B (Methodological)},
  volume  = {58},
  number  = {1},
  pages   = {267--288},
  year    = {1996},
  doi     = {10.1111/j.2517-6161.1996.tb02080.x}
}
```
