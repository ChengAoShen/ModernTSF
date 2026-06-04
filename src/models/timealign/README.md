---
model: "TimeAlign"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/TimeAlign.toml"
registry: "models.timealign.registry"
paper_title: "Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting"
venue: "ICLR 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2509.14181"
---
# TimeAlign

TimeAlign is a lightweight, plug-and-play framework for time series forecasting that aligns past and future representations to bridge the distributional gap between historical inputs and future targets. It establishes a new representation paradigm by aligning auxiliary features via a reconstruction task and feeding them back into any base forecaster, with gains arising primarily from correcting frequency mismatches between historical inputs and future outputs.

## Paper
- **Title**: Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting
- **Venue**: ICLR 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.14181

## Abstract
Although contrastive and other representation-learning methods have long been explored in vision and NLP, their adoption in modern time series forecasters remains limited. We believe they hold strong promise for this domain. To unlock this potential, we explicitly align past and future representations, thereby bridging the distributional gap between input histories and future targets. To this end, we introduce TimeAlign, a lightweight, plug-and-play framework that establishes a new representation paradigm, distinct from contrastive learning, by aligning auxiliary features via a simple reconstruction task and feeding them back into any base forecaster. Extensive experiments across eight benchmarks verify its superior performance. Further studies indicate that the gains arise primarily from correcting frequency mismatches between historical inputs and future outputs. Additionally, we provide two theoretical justifications for how reconstruction improves forecasting generalization and how alignment increases the mutual information between learned representations and predicted targets. The code is available at https://github.com/TROUBADOUR000/TimeAlign.

## In ModernTSF
Default config: `configs/models/TimeAlign.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
