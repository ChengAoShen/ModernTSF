---
model: "xPatch"
forecasting_setting: "time_series"
config: "configs/models/xPatch.toml"
registry: "models.xpatch.registry"
paper_title: "xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition"
venue: "AAAI 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2412.17323"
---
# xPatch

xPatch is a dual-stream time series forecasting model that combines an exponential seasonal-trend decomposition module with two parallel processing streams — an MLP-based linear stream and a CNN-based non-linear stream — both using patch-based channel-independent representations, and further employs a robust arctangent loss function and a sigmoid learning rate schedule to prevent overfitting.

## Paper
- **Title**: xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2412.17323

## Abstract
In recent years, the application of transformer-based models in time-series forecasting has received significant attention. While often demonstrating promising results, the transformer architecture encounters challenges in fully exploiting the temporal relations within time series data due to its attention mechanism. In this work, we design eXponential Patch (xPatch for short), a novel dual-stream architecture that utilizes exponential decomposition. Inspired by the classical exponential smoothing approaches, xPatch introduces the innovative seasonal-trend exponential decomposition module. Additionally, we propose a dual-flow architecture that consists of an MLP-based linear stream and a CNN-based non-linear stream. This model investigates the benefits of employing patching and channel-independence techniques within a non-transformer model. Finally, we develop a robust arctangent loss function and a sigmoid learning rate adjustment scheme, which prevent overfitting and boost forecasting performance.

## In ModernTSF
Default config: `configs/models/xPatch.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
