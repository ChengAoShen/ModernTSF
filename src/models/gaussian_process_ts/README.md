---
model: "GaussianProcessTS"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/GaussianProcessTS.toml"
registry: "models.gaussian_process_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# GaussianProcessTS

GaussianProcessTS is a classical statistical baseline for multivariate and univariate time-series forecasting. It is implemented as a PyTorch-native adapter (MLTSFModel, family="gaussian_process") that wraps a Gaussian Process-inspired prototype-kernel predictor: a learnable set of prototype embeddings are matched against encoded input windows via a soft-attention kernel, and the weighted prototype outputs form the forecast. The model runs on CPU, CUDA, or MPS through the standard ModernTSF trainer interface.

## Paper
- **Title**: N/A — classical Gaussian Process regression (no single defining paper)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Gaussian Process (GP) regression is a non-parametric Bayesian approach to supervised learning that places a prior distribution over functions and uses kernel functions to measure similarity between inputs. Given training observations, the GP posterior provides closed-form mean predictions and uncertainty estimates. Key design choices are the choice of covariance (kernel) function — common options include the squared-exponential (RBF), Matérn, and periodic kernels — and the noise model. The ModernTSF adapter distills this principle into a differentiable prototype-kernel module: a bank of learnable prototypes is queried via a scaled-dot-product kernel over encoded input windows, and the aggregated prototype responses produce the multi-step forecast, enabling GPU-accelerated training through standard backpropagation.

## In ModernTSF
Default config: `configs/models/GaussianProcessTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
