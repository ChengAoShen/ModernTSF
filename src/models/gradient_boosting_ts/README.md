---
name: "GradientBoostingTS"
implementation: rewrite
summary: "GradientBoostingTS is a PyTorch-native adapter that applies gradient boosting regression to multivariate time series forecasting. It uses a residual ensemble of soft decision trees with linear base learners, trained end-to-end through the standard ModernTSF trainer, and can operate on CPU, CUDA, or MPS devices."
paper:
  title: "Greedy function approximation: A gradient boosting machine"
  venue: "Annals of Statistics, 2001"
  year: 2001
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# GradientBoostingTS

GradientBoostingTS is a PyTorch-native adapter that applies gradient boosting regression to multivariate time series forecasting. It uses a residual ensemble of soft decision trees with linear base learners, trained end-to-end through the standard ModernTSF trainer, and can operate on CPU, CUDA, or MPS devices.

## Paper
- **Title**: Greedy function approximation: A gradient boosting machine
- **Venue**: Annals of Statistics, 2001
- **Published**: 2001
- **arXiv**: N/A

## Abstract
Function estimation/approximation is viewed from the perspective of numerical optimization in function space, rather than parameter space. A connection between stagewise additive expansions and steepest-descent minimization is identified. A general gradient descent "boosting" paradigm is developed for additive expansions based on any fitting criterion. Special enhancements are derived for regression with squared error loss, absolute error loss, and huberized M-loss, with applications to least-squares, least absolute deviation, and Huber-M loss functions for regression, and multiclass logistic likelihood for classification. Regression trees are shown to be especially amenable to this approach, giving rise to the Gradient Tree Boosting procedure. Competitive statistical performance of the resulting procedures is demonstrated on several datasets, producing highly robust, interpretable nonparametric regression and classification models appropriate for data mining applications.

## In ModernTSF
Default config: `configs/models/GradientBoostingTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@article{friedman2001greedy,
  author  = {Jerome H. Friedman},
  title   = {Greedy Function Approximation: A Gradient Boosting Machine},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232},
  year    = {2001},
  doi     = {10.1214/aos/1013203451}
}
```
