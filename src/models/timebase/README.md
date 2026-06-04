---
model: "TimeBase"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/TimeBase.toml"
registry: "models.timebase.registry"
paper_title: "TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting"
venue: "ICML 2025"
year: 2025
arxiv: ""
---
# TimeBase

TimeBase is an ultra-lightweight network for long-term time series forecasting that extracts core basis temporal components from the input window and transforms traditional point-level prediction into efficient segment-level forecasting, exploiting the temporal pattern similarity and low-rank structure inherent in long-horizon time series data.

## Paper
- **Title**: TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting
- **Venue**: ICML 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Long-term time series forecasting (LTSF) has traditionally relied on large parameters to capture extended temporal dependencies, resulting in substantial computational costs and inefficiencies in both memory usage and processing time. However, time series data, unlike high-dimensional images or text, often exhibit temporal pattern similarity and low-rank structures, especially in long-term horizons. By leveraging this structure, models can be guided to focus on more essential, concise temporal data, improving both accuracy and computational efficiency. In this paper, we introduce TimeBase, an ultra-lightweight network to harness the power of minimalism in LTSF. TimeBase 1) extracts core basis temporal components and 2) transforms traditional point-level forecasting into efficient segment-level forecasting, achieving optimal utilization of both data and parameters. Extensive experiments on diverse real-world datasets show that TimeBase achieves remarkable efficiency and secures competitive forecasting performance. Additionally, TimeBase can also serve as a very effective plug-and-play complexity reducer for any patch-based forecasting models.

## In ModernTSF
Default config: `configs/models/TimeBase.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
