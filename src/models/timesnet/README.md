---
model: "TimesNet"
forecasting_setting: "time_series"
config: "configs/models/TimesNet.toml"
registry: "models.timesnet.registry"
paper_title: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"
venue: "ICLR 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2210.02186"
---
# TimesNet

TimesNet is a task-general time series analysis backbone for the standard time-series forecasting setting. It observes that real-world time series exhibit multi-periodicity, then transforms the 1D sequence into a set of 2D tensors (one per detected period) so that intraperiod and interperiod variations map to columns and rows respectively — enabling powerful 2D vision-style convolution kernels (via a parameter-efficient inception block) to model complex temporal patterns that are difficult to capture in 1D.

## Paper
- **Title**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
- **Venue**: ICLR 2023
- **Published**: 2023 (arXiv: 2022-10)
- **arXiv**: https://arxiv.org/abs/2210.02186

## Abstract
Time series analysis is of immense importance in extensive applications, such as weather forecasting, anomaly detection, and action recognition. This paper focuses on temporal variation modeling, which is the common key problem of extensive analysis tasks. Previous methods attempt to accomplish this directly from the 1D time series, which is extremely challenging due to the intricate temporal patterns. Based on the observation of multi-periodicity in time series, we ravel out the complex temporal variations into the multiple intraperiod- and interperiod-variations. To tackle the limitations of 1D time series in representation capability, we extend the analysis of temporal variations into the 2D space by transforming the 1D time series into a set of 2D tensors based on multiple periods. This transformation can embed the intraperiod- and interperiod-variations into the columns and rows of the 2D tensors respectively, making the 2D-variations to be easily modeled by 2D kernels. Technically, we propose the TimesNet with TimesBlock as a task-general backbone for time series analysis. TimesBlock can discover the multi-periodicity adaptively and extract the complex temporal variations from transformed 2D tensors by a parameter-efficient inception block. Our proposed TimesNet achieves consistent state-of-the-art in five mainstream time series analysis tasks, including short- and long-term forecasting, imputation, classification, and anomaly detection.

## In ModernTSF
Default config: `configs/models/TimesNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
