---
model: "MAFS"
forecasting_setting: "time_series"
config: "configs/models/MAFS.toml"
registry: "models.mafs.registry"
paper_title: "Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation"
venue: "NeurIPS 2025"
year: 2025
arxiv: ""
---
# MAFS

MAFS (Multi-Agent Forecasting System) is a time series forecasting framework that replaces the conventional single-model paradigm with a cooperative system of specialized agents. The forecasting task is decomposed into multiple sub-tasks — covering different temporal perspectives such as varying resolutions or signal characteristics — each handled by a dedicated iTransformer-based agent. Agents communicate through learnable topology graphs (ring, star, chain, or fully connected), and a lightweight voting aggregator integrates their outputs into the final prediction for each channel.

## Paper
- **Title**: Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Time series forecasting is a critical and complex task, characterized by diverse temporal patterns, varying statistical properties, and different prediction horizons across datasets and domains. Conventional approaches typically rely on a single, unified model architecture to handle all forecasting scenarios, but such monolithic models struggle to generalize across dynamically evolving time series with shifting patterns. In this paper, we propose a Multi-Agent Forecasting System (MAFS) that abandons the one-size-fits-all paradigm by decomposing the forecasting task into multiple sub-tasks, each handled by a dedicated agent trained on specific temporal perspectives. Agents share and refine information through different communication topology, enabling cooperative reasoning across different temporal views, and a lightweight voting aggregator then integrates their outputs into consistent final predictions. Extensive experiments across 11 benchmarks demonstrate that MAFS significantly outperforms traditional single-model approaches, yielding more robust and adaptable forecasts.

## In ModernTSF
Default config: `configs/models/MAFS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

The official project does not currently publish a paper BibTeX entry or a
stable proceedings identifier. Until one is available, cite the official
software repository without inventing paper metadata:

```bibtex
@misc{mafs2025software,
  author       = {{MAFS Contributors}},
  title        = {Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/h505023992/MAFS}
}
```
