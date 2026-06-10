---
model: "ElasticNetTS"
forecasting_setting: "time_series"
config: "configs/models/ElasticNetTS.toml"
registry: "models.elastic_net_ts.registry"
paper_title: "Regularization and Variable Selection via the Elastic Net"
venue: "Journal of the Royal Statistical Society, Series B"
year: 2005
arxiv: ""
---
# ElasticNetTS

ElasticNetTS is a time series forecasting model that applies the Elastic Net regression method — a linear predictor combining L1 (Lasso) and L2 (Ridge) regularization — to autoregressive lag-feature forecasting. It fits one linear model per channel and output step, making it an interpretable and computationally efficient baseline. The ModernTSF adapter wraps the Elastic Net as a `torch.nn.Module` so it runs within the standard training loop and can be dispatched to CUDA/MPS devices.

## Paper
- **Title**: Regularization and Variable Selection via the Elastic Net
- **Venue**: Journal of the Royal Statistical Society, Series B
- **Published**: 2005
- **arXiv**: N/A

## Abstract
Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalty terms of the Lasso and Ridge methods. It was introduced by Zou and Hastie (2005) to address the limitations of Lasso — in particular its instability when features are correlated and its inability to select more variables than observations. The Elastic Net penalty encourages a grouping effect in which strongly correlated predictors tend to be selected or dropped together. This combination achieves the sparsity of Lasso and the stability of Ridge, making it well suited to high-dimensional regression and variable selection problems where predictors exhibit correlation structure.

## In ModernTSF
Default config: `configs/models/ElasticNetTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{zou2005elasticnet,
  author  = {Hui Zou and Trevor Hastie},
  title   = {Regularization and Variable Selection via the Elastic Net},
  journal = {Journal of the Royal Statistical Society: Series {B} (Statistical Methodology)},
  volume  = {67},
  number  = {2},
  pages   = {301--320},
  year    = {2005},
  doi     = {10.1111/j.1467-9868.2005.00503.x}
}
```
