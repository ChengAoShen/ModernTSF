---
model: "RandomForestTS"
forecasting_setting: "time_series"
config: "configs/models/RandomForestTS.toml"
registry: "models.random_forest_ts.registry"
paper_title: "Random Forests"
venue: "Machine Learning, 2001"
year: 2001
arxiv: ""
---
# RandomForestTS

RandomForestTS is a PyTorch-native adapter that applies the random forest ensemble strategy to multivariate time series forecasting. It implements a differentiable soft-tree ensemble — multiple randomized decision trees whose outputs are averaged — operating on lagged input windows, and runs through the standard ModernTSF trainer on CPU, CUDA, or MPS devices.

## Paper
- **Title**: Random Forests
- **Venue**: Machine Learning, 2001
- **Published**: 2001
- **arXiv**: N/A

## Abstract
Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost, but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.

## In ModernTSF
Default config: `configs/models/RandomForestTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{breiman2001random,
  author  = {Leo Breiman},
  title   = {Random Forests},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001},
  doi     = {10.1023/A:1010933404324}
}
```
