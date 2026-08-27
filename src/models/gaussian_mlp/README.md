---
name: "GaussianMLP"
implementation: rewrite
summary: "GaussianMLP is a simple **parametric probabilistic** baseline: an MLP maps the flattened input window to per-step Gaussian parameters `(loc, scale)` for every horizon step and channel, returning `(B, pred_len, C, 2)` with a strictly positive scale (`softplus + eps`). It is trained by maximum likelihood (`nll_gaussian`) and scored with the closed-form Gaussian CRPS plus coverage / width. It serves as the minimal reference for the `distribution` output type — the parametric counterpart to the quantile models."
paper:
  title: "Gaussian-head MLP (ModernTSF parametric probabilistic baseline)"
  venue: "ModernTSF"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# GaussianMLP

GaussianMLP is a simple **parametric probabilistic** baseline: an MLP maps the
flattened input window to per-step Gaussian parameters `(loc, scale)` for every
horizon step and channel, returning `(B, pred_len, C, 2)` with a strictly
positive scale (`softplus + eps`). It is trained by maximum likelihood
(`nll_gaussian`) and scored with the closed-form Gaussian CRPS plus
coverage / width. It serves as the minimal reference for the `distribution`
output type — the parametric counterpart to the quantile models.

## Method
A standard parametric forecasting baseline (Gaussian likelihood head, as
popularized by DeepAR-style models). No single canonical paper; this is a
ModernTSF reference implementation of the `distribution` output axis.

## In ModernTSF
`output_type = "distribution"`, `distribution_family = "gaussian"`; pair with
`[training] loss = "nll_gaussian"`. Default config:
`configs/models/GaussianMLP.toml`; specification: `spec.py`; implementation:
`model.py`. See the `deepar` model for an RNN-based distribution forecaster.

## Source and verification

- Implementation: `rewrite` (clean-room audit pending). This is an intentional in-repository baseline, not an external paper reproduction.
- It predicts independent Gaussian location/scale pairs; cross-channel and cross-horizon covariance are not modeled.
