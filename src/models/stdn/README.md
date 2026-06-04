---
model: "STDN"
forecasting_setting: "spatiotemporal"
config: "configs/models/STDN.toml"
registry: "models.stdn.registry"
paper_title: "Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting"
venue: "AAAI 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2502.12213"
---
# STDN

STDN is a spatiotemporal learning model for node-structured graph data. It constructs a dynamic graph to represent traffic flow and captures global dynamics through novel spatio-temporal embeddings, then applies a trend-seasonality decomposition module to disentangle trend-cyclical and seasonal components for each node, before passing them through an encoder-decoder network.

## Paper
- **Title**: Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2025-02)
- **arXiv**: https://arxiv.org/abs/2502.12213

## Abstract
Traffic prediction is critical for optimizing travel scheduling and enhancing public safety, yet the complex spatial and temporal dynamics within traffic data present significant challenges for accurate forecasting. In this paper, we introduce a novel model, the Spatiotemporal-aware Trend-Seasonality Decomposition Network (STDN). This model begins by constructing a dynamic graph structure to represent traffic flow and incorporates novel spatio-temporal embeddings to jointly capture global traffic dynamics. The representations learned are further refined by a specially designed trend-seasonality decomposition module, which disentangles the trend-cyclical component and seasonal component for each traffic node at different times within the graph. These components are subsequently processed through an encoder-decoder network to generate the final predictions. Extensive experiments conducted on real-world traffic datasets demonstrate that STDN achieves superior performance with remarkable computation cost. Furthermore, we have released a new traffic dataset named JiNan, which features unique inner-city dynamics, thereby enriching the scenario comprehensiveness in traffic prediction evaluation.

## In ModernTSF
Default config: `configs/models/STDN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
