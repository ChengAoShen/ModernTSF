---
name: "CRIB"
implementation: rewrite
summary: "CRIB is a forecasting port of a missing-value TSF architecture. In ModernTSF it trains on complete standard forecasting windows (the upstream missing-value data pipeline is not included). It patches the input, encodes it with a TCN + unified-variate Transformer into an Information-Bottleneck latent, and predicts with a small MLP head. A consistency regularizer aligns the representations of the clean input and a noisy second view, while an IB (KL) term compresses the latent — together filtering the noise that missing values inject."
paper:
  title: "Revisiting Multivariate Time Series Forecasting with Missing Values"
  venue: "ICLR 2026"
  year: 2026
  url: "https://arxiv.org/abs/2509.23494"
codebase:
  url: "https://github.com/Muyiiiii/CRIB"
  revision: "a457672c7b0152f74c929858dba2a9c886405519"
  license: "NOASSERTION"
  usage: reference-only
---
# CRIB

CRIB is a forecasting port of a missing-value TSF architecture. In ModernTSF it
trains on complete standard forecasting windows (the upstream missing-value data
pipeline is not included). It patches the input, encodes it with a TCN +
unified-variate Transformer into an Information-Bottleneck latent, and predicts
with a small MLP head. A consistency regularizer aligns the representations of
the clean input and a noisy second view, while an IB (KL) term compresses the
latent — together filtering the noise that missing values inject.

<!-- model-card:canonical:start -->
## Method overview

CRIB is a forecasting port of a missing-value TSF architecture.

## Core architecture

In ModernTSF it trains on complete standard forecasting windows (the upstream missing-value data pipeline is not included). It patches the input, encodes it with a TCN + unified-variate Transformer into an Information-Bottleneck latent, and predicts with a small MLP head. A consistency regularizer aligns the representations of the clean input and a noisy second view, while an IB (KL) term compresses the latent — together filtering the noise that missing values inject.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2509.23494); title: Revisiting Multivariate Time Series Forecasting with Missing Values; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/Muyiiiii/CRIB); revision: `a457672c7b0152f74c929858dba2a9c886405519`; license: `NOASSERTION`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/CRIB.toml`](../../../configs/models/CRIB.toml).

## Differences

Compared with the author repository at `a457672c7b0152f74c929858dba2a9c886405519`. The missing-value data pipeline is absent and the repository has no explicit code license, so this model remains pending implementation audit.

## Shared components

- [`revin`](../../components/revin.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `patch_len=8`, `model_dim=32`, `heads_num=4`, `enc_num=3`, `dropout=0.1`, `activation='relu'`, `consis_weight=1.0`, `kl_weight=1e-06`
<!-- model-card:canonical:end -->

## Training objective
`L = IB_weight · MAE(Ŷ, Y) + Consis_weight · MSE(enc_clean, enc_noisy) + KL_weight · KL(q(z|x)‖N(0,I))`
(defaults `IB_weight=1`, `Consis_weight=1`, `KL_weight=1e-6`).

## In ModernTSF
Default config: `configs/models/CRIB.toml`; specification: `spec.py`; implementation:
`model.py`.

**Model-only port** (per request): the upstream missing-value masking /
augmentation **data pipeline is NOT included** — CRIB trains on the standard
complete forecasting windows (equivalent to upstream `missing_rate=0`). The
vendored core reproduces the upstream architecture (a patching adapter maps the
`(B, seq_len, enc_in)` input to the patched 4-D tensor CRIB expects; dead/unused
upstream submodules are dropped). The consistency + KL terms are computed inside
`forward` from the input alone and exposed via the trainer's `aux_loss`
convention; the MAE prediction term is the configured `training.loss` (use
`mae`). Constraints: `patch_len` must divide `seq_len`, and `model_dim` must be
divisible by `heads_num`. Verify with
`uv run tsf smoke --model CRIB`.

Upstream reference: https://github.com/Muyiiiii/CRIB

## Source and verification

Compared with the author repository at `a457672c7b0152f74c929858dba2a9c886405519`. The missing-value data pipeline is absent and the repository has no explicit code license, so this model remains pending implementation audit.
