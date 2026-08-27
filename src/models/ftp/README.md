---
name: "FTP"
implementation: rewrite
summary: "FTP (FusionTimePatch) is a Transformer-based multivariate time-series forecasting model for the standard time-series forecasting setting. It unifies channel-independent and channel-mixing views through a multi-scale patch recursion strategy that simultaneously captures local temporal patterns and global inter-channel dependencies, combining a Dual-GLF component with a channel-enhancement module."
paper:
  title: "Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting"
  venue: "AAAI 2026"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# FTP

FTP (FusionTimePatch) is a Transformer-based multivariate time-series forecasting model for the standard time-series forecasting setting. It unifies channel-independent and channel-mixing views through a multi-scale patch recursion strategy that simultaneously captures local temporal patterns and global inter-channel dependencies, combining a Dual-GLF component with a channel-enhancement module.

<!-- model-card:canonical:start -->
## Method overview

FTP (FusionTimePatch) is a Transformer-based multivariate time-series forecasting model for the standard time-series forecasting setting.

## Core architecture

It unifies channel-independent and channel-mixing views through a multi-scale patch recursion strategy that simultaneously captures local temporal patterns and global inter-channel dependencies, combining a Dual-GLF component with a channel-enhancement module.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting; venue/year: AAAI 2026 / 2026
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/FTP.toml`](../../../configs/models/FTP.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
The official paper abstract is not available on arXiv. According to the upstream repository (https://github.com/Zhveh7/FTP), FTP introduces three core components: Dual-GLF, which introduces channel-independent (CI) and channel-mixing (CM) perspectives in parallel, leveraging multi-scale patch recursion to capture both local and global temporal patterns; a Channel Enhancement (CE) module that enhances salient channel features and diffuses them across channels, improving sensitivity to anomalies and underlying drivers; and a hierarchical patch recursion mechanism that aggregates patch representations across scales to build a rich global-local representation. The model achieves competitive performance on standard long-term forecasting benchmarks.

## In ModernTSF
Default config: `configs/models/FTP.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/ZhangZBZGJSP26,
  author       = {Wenhao Zhang and
                  Chun Zhang and
                  Wei Bai and
                  Ning Zhang and
                  Changxia Gao and
                  Yuxin Jia and
                  Chenhao Shi and
                  Shaoxiong Pang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion
                  for Global-Local Representation Synergy in Multivariate Time Series
                  Forecasting},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {28427--28436},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i33.40072},
  doi          = {10.1609/AAAI.V40I33.40072},
  timestamp    = {Tue, 24 Mar 2026 17:03:54 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/ZhangZBZGJSP26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
