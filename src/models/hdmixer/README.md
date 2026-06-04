---
model: "HDMixer"
forecasting_setting: "time_series"
config: "configs/models/HDMixer.toml"
registry: "models.hdmixer.registry"
paper_title: "HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting"
venue: "AAAI 2024"
year: 2024
arxiv: ""
---
# HDMixer

HDMixer is a pure MLP-based time series forecasting model for multivariate prediction that addresses two limitations of standard patch-based approaches: fixed-length patches lose temporal boundary information (e.g., peaks and periods are cut arbitrarily), and existing methods focus mainly on long-range cross-patch dependencies while ignoring short-range within-patch and cross-variable interactions. HDMixer introduces a Length-Extendable Patcher (LEP) to enrich patch boundary information and a Hierarchical Dependency Explorer (HDE) that models all three dependency levels — within-patch (short-term), across-patch (long-term), and cross-variable — using pure MLPs.

## Paper
- **Title**: HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Multivariate time series (MTS) prediction has been widely adopted in various scenarios. Recently, some methods have employed patching to enhance local semantics and improve model performance. However, length-fixed patch are prone to losing temporal boundary information, such as complete peaks and periods. Moreover, existing methods mainly focus on modeling long-term dependencies across patches, while paying little attention to other dimensions (e.g., short-term dependencies within patches and complex interactions among cross-variavle patches). To address these challenges, we propose a pure MLP-based HDMixer, aiming to acquire patches with richer semantic information and efficiently modeling hierarchical interactions. Specifically, we design a Length-Extendable Patcher (LEP) tailored to MTS, which enriches the boundary information of patches and alleviates semantic incoherence in series. Subsequently, we devise a Hierarchical Dependency Explorer (HDE) based on pure MLPs. This explorer effectively models short-term dependencies within patches, long-term dependencies across patches, and complex interactions among variables. Extensive experiments on 9 real-world datasets demonstrate the superiority of our approach.

## In ModernTSF
Default config: `configs/models/HDMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
