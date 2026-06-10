---
model: "PCDCNet"
forecasting_setting: "covariate"
config: "configs/models/PCDCNet.toml"
registry: "models.pcdcnet.registry"
paper_title: "PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints"
venue: "arXiv preprint"
year: 2025
arxiv: "https://arxiv.org/abs/2505.19842"
---
# PCDCNet

PCDCNet is a covariate-prediction model for air quality forecasting in a node-structured spatiotemporal setting, where each node is a monitoring station. It integrates numerical modeling principles (emissions, meteorological influences, and physical-chemical domain constraints) with deep learning components — specifically graph-based spatial transport, recurrent temporal accumulation, and local interaction representation enhancement — to forecast 72-hour PM2.5 and O3 concentrations at the station level.

## Paper
- **Title**: PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.19842

## Abstract
Air quality forecasting (AQF) is critical for public health and environmental management, yet remains challenging due to the complex interplay of emissions, meteorology, and chemical transformations. Traditional numerical models, such as CMAQ and WRF-Chem, provide physically grounded simulations but are computationally expensive and rely on uncertain emission inventories. Deep learning models, while computationally efficient, often struggle with generalization due to their lack of physical constraints. To bridge this gap, we propose PCDCNet, a surrogate model that integrates numerical modeling principles with deep learning. PCDCNet explicitly incorporates emissions, meteorological influences, and domain-informed constraints to model pollutant formation, transport, and dissipation. By combining graph-based spatial transport modeling, recurrent structures for temporal accumulation, and representation enhancement for local interactions, PCDCNet achieves state-of-the-art (SOTA) performance in 72-hour station-level PM2.5 and O3 forecasting while significantly reducing computational costs. Furthermore, our model is deployed in an online platform, providing free, real-time air quality forecasts, demonstrating its scalability and societal impact. By aligning deep learning with physical consistency, PCDCNet offers a practical and interpretable solution for AQF, enabling informed decision-making for both personal and regulatory applications.

## In ModernTSF
Default config: `configs/models/PCDCNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{wang2025pcdcnet,
  author        = {Shuo Wang and
                  Yun Cheng and
                  Qingye Meng and
                  Olga Saukh and
                  Jiang Zhang and
                  Jingfang Fan and
                  Yuanting Zhang and
                  Xingyuan Yuan and
                  Lothar Thiele},
  title         = {PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints},
  year          = {2025},
  eprint        = {2505.19842},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.19842}
}
```
