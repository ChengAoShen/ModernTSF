---
model: "TimeMosaic"
forecasting_setting: "time_series"
config: "configs/models/TimeMosaic.toml"
registry: "models.timemosaic.registry"
paper_title: "TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2509.19406"
---
# TimeMosaic

TimeMosaic is a time-series forecasting model designed to handle temporal heterogeneity in multivariate data. It employs adaptive patch embedding to dynamically adjust segmentation granularity based on local information density, and a segment-wise decoder that treats each prediction horizon as a related but distinct sub-task, adapting to horizon-specific difficulty rather than applying a single uniform decoder.

## Paper
- **Title**: TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.19406

## Abstract
Multivariate time series forecasting is essential in domains such as finance, transportation, climate, and energy. However, existing patch-based methods typically adopt fixed-length segmentation, overlooking the heterogeneity of local temporal dynamics and the decoding heterogeneity of forecasting. Such designs lose details in information-dense regions, introduce redundancy in stable segments, and fail to capture the distinct complexities of short-term and long-term horizons. We propose TimeMosaic, a forecasting framework that aims to address temporal heterogeneity. TimeMosaic employs adaptive patch embedding to dynamically adjust granularity according to local information density, balancing motif reuse with structural clarity while preserving temporal continuity. In addition, it introduces segment-wise decoding that treats each prediction horizon as a related subtask and adapts to horizon-specific difficulty and information requirements, rather than applying a single uniform decoder. Extensive evaluations on benchmark datasets demonstrate that TimeMosaic delivers consistent improvements over existing methods, and our model trained on the large-scale corpus with 321 billion observations achieves performance competitive with state-of-the-art TSFMs.

## In ModernTSF
Default config: `configs/models/TimeMosaic.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
