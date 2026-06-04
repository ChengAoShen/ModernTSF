---
model: "DecisionTreeTS"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/DecisionTreeTS.toml"
registry: "models.decision_tree_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# DecisionTreeTS

DecisionTreeTS is a PyTorch-native adapter that wraps a decision-tree-style predictor over flattened lag features for univariate and multivariate time series forecasting. It registers under the standard ModernTSF trainer interface, allowing the tree-based computation to run on CPU, CUDA, or MPS tensors.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Decision trees are classical non-parametric supervised learning models that recursively partition the input feature space using axis-aligned splits, selecting the split at each node by minimising an impurity criterion (e.g., mean squared error for regression). For time series forecasting, the model is applied by constructing a feature matrix of lagged input values and training a separate tree (or a single multi-output tree) to predict each future step. Although decision trees are highly interpretable and require no gradient-based optimisation, they can overfit without regularisation (maximum depth, minimum samples per leaf) and do not naturally capture sequential structure. In ModernTSF they are wrapped as a differentiable-style torch.nn.Module for uniform pipeline integration.

## In ModernTSF
Default config: `configs/models/DecisionTreeTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
