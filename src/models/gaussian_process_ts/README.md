---
name: "GaussianProcessTS"
implementation: rewrite
summary: "GaussianProcessTS is a classical statistical baseline for multivariate and univariate time-series forecasting. It is implemented as a PyTorch-native adapter (MLTSFModel, family=\"gaussian_process\") that wraps a Gaussian Process-inspired prototype-kernel predictor: a learnable set of prototype embeddings are matched against encoded input windows via a soft-attention kernel, and the weighted prototype outputs form the forecast. The model runs on CPU, CUDA, or MPS through the standard ModernTSF trainer interface."
paper:
  title: ""
  venue: "N/A (classical baseline)"
  year: null
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# GaussianProcessTS

GaussianProcessTS is a classical statistical baseline for multivariate and univariate time-series forecasting. It is implemented as a PyTorch-native adapter (MLTSFModel, family="gaussian_process") that wraps a Gaussian Process-inspired prototype-kernel predictor: a learnable set of prototype embeddings are matched against encoded input windows via a soft-attention kernel, and the weighted prototype outputs form the forecast. The model runs on CPU, CUDA, or MPS through the standard ModernTSF trainer interface.

<!-- model-card:canonical:start -->
## Method overview

GaussianProcessTS is a classical statistical baseline for multivariate and univariate time-series forecasting.

## Core architecture

It is implemented as a PyTorch-native adapter (MLTSFModel, family="gaussian_process") that wraps a Gaussian Process-inspired prototype-kernel predictor: a learnable set of prototype embeddings are matched against encoded input windows via a soft-attention kernel, and the weighted prototype outputs form the forecast. The model runs on CPU, CUDA, or MPS through the standard ModernTSF trainer interface.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: not available; venue/year: N/A (classical baseline) / not available
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/GaussianProcessTS.toml`](../../../configs/models/GaussianProcessTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=48`, `kernel_gamma=0.04`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: N/A — classical Gaussian Process regression (no single defining paper)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Gaussian Process (GP) regression is a non-parametric Bayesian approach to supervised learning that places a prior distribution over functions and uses kernel functions to measure similarity between inputs. Given training observations, the GP posterior provides closed-form mean predictions and uncertainty estimates. Key design choices are the choice of covariance (kernel) function — common options include the squared-exponential (RBF), Matérn, and periodic kernels — and the noise model. The ModernTSF adapter distills this principle into a differentiable prototype-kernel module: a bank of learnable prototypes is queried via a scaled-dot-product kernel over encoded input windows, and the aggregated prototype responses produce the multi-step forecast, enabling GPU-accelerated training through standard backpropagation.

## In ModernTSF
Default config: `configs/models/GaussianProcessTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@book{DBLP:books/lib/RasmussenW06,
  author       = {Carl Edward Rasmussen and
                  Christopher K. I. Williams},
  title        = {Gaussian processes for machine learning},
  series       = {Adaptive computation and machine learning},
  publisher    = {{MIT} Press},
  year         = {2006},
  url          = {https://www.worldcat.org/oclc/61285753},
  isbn         = {026218253X},
  timestamp    = {Fri, 17 Jul 2020 16:12:42 +0200},
  biburl       = {https://dblp.org/rec/books/lib/RasmussenW06.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
