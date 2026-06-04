---
model: "TimeKAN"
category: "time_series"
category_name: "Time Series"
forecasting_setting: "time_series"
config: "configs/models/TimeKAN.toml"
registry: "models.timekan.registry"
paper_title: "TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting"
venue: "arXiv preprint"
year: 2025
arxiv: "https://arxiv.org/abs/2502.06910"
---
# TimeKAN

TimeKAN is a time series forecasting model that combines Kolmogorov-Arnold Networks (KANs) with multi-scale frequency decomposition. It decomposes a mixed-frequency input series into individual frequency bands via Cascaded Frequency Decomposition (CFD) blocks, learns band-specific temporal patterns with Multi-order KAN Representation Learning (M-KAN) blocks that exploit the flexibility of KANs, and recombines the bands via Frequency Mixing blocks to produce accurate multi-horizon predictions. The architecture achieves state-of-the-art results while remaining extremely lightweight.

## Paper
- **Title**: TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-02)
- **arXiv**: https://arxiv.org/abs/2502.06910

## Abstract
Real-world time series often have multiple frequency components that are intertwined with each other, making accurate time series forecasting challenging. Decomposing the mixed frequency components into multiple single frequency components is a natural choice. However, the information density of patterns varies across different frequencies, and employing a uniform modeling approach for different frequency components can lead to inaccurate characterization. To address this challenges, inspired by the flexibility of the recent Kolmogorov-Arnold Network (KAN), we propose a KAN-based Frequency Decomposition Learning architecture (TimeKAN) to address the complex forecasting challenges caused by multiple frequency mixtures. Specifically, TimeKAN mainly consists of three components: Cascaded Frequency Decomposition (CFD) blocks, Multi-order KAN Representation Learning (M-KAN) blocks and Frequency Mixing blocks. CFD blocks adopt a bottom-up cascading approach to obtain series representations for each frequency band. Benefiting from the high flexibility of KAN, we design a novel M-KAN block to learn and represent specific temporal patterns within each frequency band. Finally, Frequency Mixing blocks is used to recombine the frequency bands into the original format. Extensive experimental results across multiple real-world time series datasets demonstrate that TimeKAN achieves state-of-the-art performance as an extremely lightweight architecture.

## In ModernTSF
Default config: `configs/models/TimeKAN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
