---
model: "CATS"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/CATS.toml"
registry: "models.cats.registry"
paper_title: "Are Self-Attentions Effective for Time Series Forecasting?"
venue: "NeurIPS 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2405.16877"
---
# CATS

CATS (Cross-Attention-only Time Series transformer) is a multivariate time series forecasting model that eliminates self-attention entirely from the Transformer architecture and relies solely on cross-attention mechanisms, using future horizon-dependent parameters as queries with enhanced parameter sharing to improve long-term forecasting accuracy while reducing parameter count and memory usage.

## Paper
- **Title**: Are Self-Attentions Effective for Time Series Forecasting?
- **Venue**: NeurIPS 2024
- **Published**: 2024 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.16877

## Abstract
Time series forecasting is crucial for applications across multiple domains and various scenarios. Although Transformer models have dramatically advanced the landscape of forecasting, their effectiveness remains debated. Recent findings have indicated that simpler linear models might outperform complex Transformer-based approaches, highlighting the potential for more streamlined architectures. In this paper, we shift the focus from evaluating the overall Transformer architecture to specifically examining the effectiveness of self-attention for time series forecasting. To this end, we introduce a new architecture, Cross-Attention-only Time Series transformer (CATS), that rethinks the traditional Transformer framework by eliminating self-attention and leveraging cross-attention mechanisms instead. By establishing future horizon-dependent parameters as queries and enhanced parameter sharing, our model not only improves long-term forecasting accuracy but also reduces the number of parameters and memory usage. Extensive experiment across various datasets demonstrates that our model achieves superior performance with the lowest mean squared error and uses fewer parameters compared to existing models.

## In ModernTSF
Default config: `configs/models/CATS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
