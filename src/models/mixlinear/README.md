---
name: "MixLinear"
implementation: rewrite
summary: "MixLinear is an ultra-lightweight multivariate time-series forecasting model for the standard time-series forecasting setting. It mixes time-domain linear projections (both intra-segment and inter-segment) with frequency-domain linear projections over a low-dimensional latent space, reducing the parameter scale of the core linear layers from O(n²) to O(n) while retaining competitive accuracy — making it well suited for resource-constrained deployment."
paper:
  title: "MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters"
  venue: "ICLR 2026"
  year: 2026
  url: "https://arxiv.org/abs/2410.02081"
codebase:
  url: "https://github.com/aitianma/MixLinear"
  revision: "42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106"
  license: "NOASSERTION"
  usage: reference-only
---
# MixLinear

MixLinear is an ultra-lightweight multivariate time-series forecasting model for the standard time-series forecasting setting. It mixes time-domain linear projections (both intra-segment and inter-segment) with frequency-domain linear projections over a low-dimensional latent space, reducing the parameter scale of the core linear layers from O(n²) to O(n) while retaining competitive accuracy — making it well suited for resource-constrained deployment.

<!-- model-card:canonical:start -->
## Method overview

MixLinear is an ultra-lightweight multivariate time-series forecasting model for the standard time-series forecasting setting.

## Core architecture

It mixes time-domain linear projections (both intra-segment and inter-segment) with frequency-domain linear projections over a low-dimensional latent space, reducing the parameter scale of the core linear layers from O(n²) to O(n) while retaining competitive accuracy — making it well suited for resource-constrained deployment.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/2410.02081); title: MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters; venue/year: ICLR 2026 / 2026
- [codebase](https://github.com/aitianma/MixLinear); revision: `42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106`; license: `NOASSERTION`; usage: `reference-only`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/MixLinear.toml`](../../../configs/models/MixLinear.toml).

## Differences

- Author source: https://github.com/aitianma/MixLinear at `42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106`; the repository declares no code license.
Implementation: **rewrite** (clean-room audit pending). No numerical parity result is recorded and source reuse cannot be certified.
- Blocker: the local time branch uses sequential `com_len` compression rather than the pinned source's square-factorized intra/inter-segment transforms; convolution and segment handling also differ materially. An unused full segment projection has been removed.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `period_len=24`, `com_len=4`, `lpf=1`, `alpha=0.5`
<!-- model-card:canonical:end -->

## Paper
- **Title**: MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2024-10)
- **arXiv**: https://arxiv.org/abs/2410.02081

## Abstract
Recently, there has been a growing interest in Long-term Time Series Forecasting (LTSF), which involves predicting long-term future values by analyzing a large amount of historical time-series data to identify patterns and trends. There exist significant challenges in LTSF due to its complex temporal dependencies and high computational demands. Although Transformer-based models offer high forecasting accuracy, they are often too compute-intensive to be deployed on devices with hardware constraints. On the other hand, the linear models aim to reduce the computational overhead by employing either decomposition methods in the time domain or compact representations in the frequency domain. In this paper, we propose MixLinear, an ultra-lightweight multivariate time series forecasting model specifically designed for resource-constrained devices. MixLinear effectively captures both temporal and frequency domain features by modeling intra-segment and inter-segment variations in the time domain and extracting frequency variations from a low-dimensional latent space in the frequency domain. By reducing the parameter scale of a downsampled n-length input/output one-layer linear model from O(n²) to O(n), MixLinear achieves efficient computation without sacrificing accuracy. Extensive evaluations with four benchmark datasets show that MixLinear attains forecasting performance comparable to, or surpassing, state-of-the-art models with significantly fewer parameters (0.1K), which makes it well-suited for deployment on devices with limited computational capacity.

## In ModernTSF
Default config: `configs/models/MixLinear.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Author source: https://github.com/aitianma/MixLinear at `42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106`; the repository declares no code license.
Implementation: **rewrite** (clean-room audit pending). No numerical parity result is recorded and source reuse cannot be certified.
- Blocker: the local time branch uses sequential `com_len` compression rather than the pinned source's square-factorized intra/inter-segment transforms; convolution and segment handling also differ materially. An unused full segment projection has been removed.

## Citation

```bibtex
@misc{ma2024mixlinear,
  author        = {Aitian Ma and
                  Dongsheng Luo and
                  Mo Sha},
  title         = {MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters},
  year          = {2024},
  eprint        = {2410.02081},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.02081}
}
```
