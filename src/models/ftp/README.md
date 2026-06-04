---
model: "FTP"
forecasting_setting: "time_series"
config: "configs/models/FTP.toml"
registry: "models.ftp.registry"
paper_title: "Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting"
venue: "AAAI 2026"
year: 2026
arxiv: ""
---
# FTP

FTP (FusionTimePatch) is a Transformer-based multivariate time-series forecasting model for the standard time-series forecasting setting. It unifies channel-independent and channel-mixing views through a multi-scale patch recursion strategy that simultaneously captures local temporal patterns and global inter-channel dependencies, combining a Dual-GLF component with a channel-enhancement module.

## Paper
- **Title**: Unifying Channel Independence and Mixing: Multi-Scale Patch Recursion for Global-Local Representation Synergy in Multivariate Time Series Forecasting
- **Venue**: AAAI 2026
- **Published**: 2026
- **arXiv**: N/A

## Abstract
The official paper abstract is not available on arXiv. According to the upstream repository (https://github.com/Zhveh7/FTP), FTP introduces three core components: Dual-GLF, which introduces channel-independent (CI) and channel-mixing (CM) perspectives in parallel, leveraging multi-scale patch recursion to capture both local and global temporal patterns; a Channel Enhancement (CE) module that enhances salient channel features and diffuses them across channels, improving sensitivity to anomalies and underlying drivers; and a hierarchical patch recursion mechanism that aggregates patch representations across scales to build a rich global-local representation. The model achieves competitive performance on standard long-term forecasting benchmarks.

## In ModernTSF
Default config: `configs/models/FTP.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
