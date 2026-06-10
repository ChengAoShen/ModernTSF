---
model: "ExtraTreesTS"
forecasting_setting: "time_series"
config: "configs/models/ExtraTreesTS.toml"
registry: "models.extra_trees_ts.registry"
paper_title: "Extremely Randomized Trees"
venue: "Machine Learning 2006"
year: 2006
arxiv: ""
---
# ExtraTreesTS

ExtraTreesTS is a time-series forecasting adapter that wraps the Extremely Randomized Trees (Extra-Trees) ensemble method inside the ModernTSF PyTorch training harness. It applies the Extra-Trees regressor — an ensemble of decision trees with randomised split thresholds — to the sliding-window forecasting task, treating each prediction horizon step as an independent regression target.

## Paper
- **Title**: Extremely Randomized Trees
- **Venue**: Machine Learning 2006
- **Published**: 2006
- **arXiv**: N/A

## Abstract
Extremely Randomized Trees (Extra-Trees) is a tree-based ensemble learning method introduced by Geurts, Ernst, and Wehenkel (2006). Like Random Forests, it builds an ensemble of unpruned decision or regression trees from the full training set, but with two key differences that increase randomisation: (1) split points are chosen uniformly at random within each feature's range rather than by optimising an impurity criterion, and (2) all training samples are used for building each tree (no bootstrap). These two choices trade a small increase in bias for a substantial reduction in variance and a significant speedup in training. The method consistently achieves competitive accuracy with Random Forests and gradient-boosted trees across regression and classification benchmarks, while being considerably faster to train.

## In ModernTSF
Default config: `configs/models/ExtraTreesTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{geurts2006extremely,
  author  = {Pierre Geurts and Damien Ernst and Louis Wehenkel},
  title   = {Extremely Randomized Trees},
  journal = {Machine Learning},
  volume  = {63},
  number  = {1},
  pages   = {3--42},
  year    = {2006},
  doi     = {10.1007/s10994-006-6226-1}
}
```
