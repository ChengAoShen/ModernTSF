---
model: "STAEformer"
forecasting_setting: "spatiotemporal"
config: "configs/models/STAEformer.toml"
registry: "models.staeformer.registry"
paper_title: "STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting"
venue: "CIKM 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2308.10425"
---
# STAEformer

STAEformer is a spatiotemporal Transformer for node-structured graph data such as traffic networks. It introduces a novel spatio-temporal adaptive embedding that jointly encodes intrinsic spatial relations between nodes and chronological temporal patterns, enabling a standard (vanilla) Transformer encoder—without complex graph convolutions—to achieve state-of-the-art performance on traffic forecasting benchmarks.

## Paper
- **Title**: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
- **Venue**: CIKM 2023
- **Published**: 2023 (arXiv: 2023-08)
- **arXiv**: https://arxiv.org/abs/2308.10425

## Abstract
With the rapid development of the Intelligent Transportation System (ITS), accurate traffic forecasting has emerged as a critical challenge. The key bottleneck lies in capturing the intricate spatio-temporal traffic patterns. In recent years, numerous neural networks with complicated architectures have been proposed to address this issue. However, the advancements in network architectures have encountered diminishing performance gains. In this study, we present a novel component called spatio-temporal adaptive embedding that can yield outstanding results with vanilla transformers. Our proposed Spatio-Temporal Adaptive Embedding transformer (STAEformer) achieves state-of-the-art performance on five real-world traffic forecasting datasets. Further experiments demonstrate that spatio-temporal adaptive embedding plays a crucial role in traffic forecasting by effectively capturing intrinsic spatio-temporal relations and chronological information in traffic time series.

## In ModernTSF
Default config: `configs/models/STAEformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{liu2023staeformer,
  author        = {Hangchen Liu and
                  Zheng Dong and
                  Renhe Jiang and
                  Jiewen Deng and
                  Jinliang Deng and
                  Quanjun Chen and
                  Xuan Song},
  title         = {STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting},
  year          = {2023},
  eprint        = {2308.10425},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2308.10425}
}
```
