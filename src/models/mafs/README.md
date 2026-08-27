---
name: "MAFS"
implementation: rewrite
summary: "MAFS (Multi-Agent Forecasting System) is a time series forecasting framework that replaces the conventional single-model paradigm with a cooperative system of specialized agents. The forecasting task is decomposed into multiple sub-tasks — covering different temporal perspectives such as varying resolutions or signal characteristics — each handled by a dedicated iTransformer-based agent. Agents communicate through learnable topology graphs (ring, star, chain, or fully connected), and a lightweight voting aggregator integrates their outputs into the final prediction for each channel."
paper:
  title: "Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation"
  venue: "NeurIPS 2025"
  year: 2025
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# MAFS

MAFS (Multi-Agent Forecasting System) is a time series forecasting framework that replaces the conventional single-model paradigm with a cooperative system of specialized agents. The forecasting task is decomposed into multiple sub-tasks — covering different temporal perspectives such as varying resolutions or signal characteristics — each handled by a dedicated iTransformer-based agent. Agents communicate through learnable topology graphs (ring, star, chain, or fully connected), and a lightweight voting aggregator integrates their outputs into the final prediction for each channel.

<!-- model-card:canonical:start -->
## Method overview

MAFS (Multi-Agent Forecasting System) is a time series forecasting framework that replaces the conventional single-model paradigm with a cooperative system of specialized agents.

## Core architecture

The forecasting task is decomposed into multiple sub-tasks — covering different temporal perspectives such as varying resolutions or signal characteristics — each handled by a dedicated iTransformer-based agent. Agents communicate through learnable topology graphs (ring, star, chain, or fully connected), and a lightweight voting aggregator integrates their outputs into the final prediction for each channel.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- paper: not available; title: Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation; venue/year: NeurIPS 2025 / 2025
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/MAFS.toml`](../../../configs/models/MAFS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `period=24`, `num_prompts=4`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Many Minds, One Goal: Time Series Forecasting via Sub-task Specialization and Inter-agent Cooperation
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Time series forecasting is a critical and complex task, characterized by diverse temporal patterns, varying statistical properties, and different prediction horizons across datasets and domains. Conventional approaches typically rely on a single, unified model architecture to handle all forecasting scenarios, but such monolithic models struggle to generalize across dynamically evolving time series with shifting patterns. In this paper, we propose a Multi-Agent Forecasting System (MAFS) that abandons the one-size-fits-all paradigm by decomposing the forecasting task into multiple sub-tasks, each handled by a dedicated agent trained on specific temporal perspectives. Agents share and refine information through different communication topology, enabling cooperative reasoning across different temporal views, and a lightweight voting aggregator then integrates their outputs into consistent final predictions. Extensive experiments across 11 benchmarks demonstrate that MAFS significantly outperforms traditional single-model approaches, yielding more robust and adaptable forecasts.

## In ModernTSF
Default config: `configs/models/MAFS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
