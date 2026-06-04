---
model: "Crossformer"
forecasting_setting: "time_series"
config: "configs/models/Crossformer.toml"
registry: "models.crossformer.registry"
paper_title: "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting"
venue: "ICLR 2023"
year: 2023
arxiv: ""
---
# Crossformer

Crossformer is a Transformer-based model for multivariate time series forecasting that explicitly models both temporal (cross-time) and inter-variable (cross-dimension) dependencies. It embeds the input series into a 2-D vector array via Dimension-Segment-Wise (DSW) embedding, applies a Two-Stage Attention (TSA) layer to efficiently capture both dependency types, and uses a Hierarchical Encoder-Decoder (HED) to leverage multi-scale temporal information for direct multi-step prediction.

## Paper
- **Title**: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
- **Venue**: ICLR 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract
Recently many deep models have been proposed for multivariate time series (MTS) forecasting. In particular, Transformer-based models have shown great potential because they can capture long-term dependency. However, existing Transformer-based models mainly focus on modeling the temporal dependency (cross-time dependency) yet often omit the dependency among different variables (cross-dimension dependency), which is critical for MTS forecasting. To fill the gap, we propose Crossformer, a Transformer-based model utilizing cross-dimension dependency for MTS forecasting. In Crossformer, the input MTS is embedded into a 2D vector array through the Dimension-Segment-Wise (DSW) embedding to preserve time and dimension information. Then the Two-Stage Attention (TSA) layer is proposed to efficiently capture the cross-time and cross-dimension dependency. Utilizing DSW embedding and TSA layer, Crossformer establishes a Hierarchical Encoder-Decoder (HED) to use the information at different scales for the final forecasting. Extensive experimental results on six real-world datasets show the effectiveness of Crossformer against previous state-of-the-arts.

## In ModernTSF
Default config: `configs/models/Crossformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
