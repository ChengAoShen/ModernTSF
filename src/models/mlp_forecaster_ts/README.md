---
model: "MLPForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/MLPForecasterTS.toml"
registry: "models.mlp_forecaster_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# MLPForecasterTS

MLPForecasterTS is a classical Multi-Layer Perceptron (MLP) baseline for time series forecasting, serving the standard univariate and multivariate prediction setting. It applies a stack of fully-connected layers with optional channel mixing and RevIN normalization to a fixed look-back window of lagged values, projecting directly to the desired forecast horizon. The model is implemented as a native PyTorch `nn.Module` adapter within the ModernTSF `_ml_tsf` family, meaning it runs on CPU, CUDA, or MPS through the standard training loop.

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
MLPForecasterTS is a foundational feedforward neural network baseline for time series forecasting. A Multi-Layer Perceptron (MLP) stacks multiple fully-connected linear layers with non-linear activations to learn a direct mapping from a fixed-length historical window of input values to a future prediction window. In the ModernTSF setting, the model operates channel-independently or with optional cross-channel mixing and applies Reversible Instance Normalization (RevIN) to stabilize training across datasets with varying scales. As a classical deep learning baseline, it serves as a simple yet non-trivial reference point for evaluating more sophisticated sequence modeling architectures.

## In ModernTSF
Default config: `configs/models/MLPForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{rumelhart1986learning,
  author  = {David E. Rumelhart and Geoffrey E. Hinton and Ronald J. Williams},
  title   = {Learning Representations by Back-Propagating Errors},
  journal = {Nature},
  volume  = {323},
  number  = {6088},
  pages   = {533--536},
  year    = {1986},
  doi     = {10.1038/323533a0}
}
```
