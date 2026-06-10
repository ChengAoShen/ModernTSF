---
model: "STOP"
forecasting_setting: "spatiotemporal"
config: "configs/models/STOP.toml"
registry: "models.stop.registry"
paper_title: "Robust Spatio-Temporal Centralized Interaction for OOD Learning"
venue: "ICML 2025"
year: 2025
arxiv: ""
---
# STOP

STOP (Spatio-Temporal OOD Processor) is a spatiotemporal forecasting model that addresses out-of-distribution generalization in graph-structured data by replacing node-to-node message passing with a centralized messaging mechanism using Context-Aware Units, combined with a message perturbation mechanism and distributionally robust optimization to produce forecasts that generalize across spatial and temporal distribution shifts.

## Paper
- **Title**: Robust Spatio-Temporal Centralized Interaction for OOD Learning
- **Venue**: ICML 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Recently, spatiotemporal graph convolutional networks have achieved dominant performance in spatiotemporal prediction tasks. However, most models relying on node-to-node messaging interaction exhibit sensitivity to spatiotemporal shifts, encountering out-of-distribution (OOD) challenges. To address these issues, we introduce Spatio-Temporal OOD Processor (STOP), which employs a centralized messaging mechanism along with a message perturbation mechanism to facilitate robust spatiotemporal interactions. Specifically, the centralized messaging mechanism integrates Context-Aware Units for coarse-grained spatiotemporal feature interactions with nodes, effectively blocking traditional node-to-node messages. We also implement a message perturbation mechanism to disrupt this messaging process, compelling the model to extract generalizable contextual features from generated variant environments. Finally, we customize a spatiotemporal distributionally robust optimization approach that exposes the model to challenging environments, thereby further enhancing its generalization capabilities. Compared with 14 baselines across six datasets, STOP achieves up to 17.01% improvement in generalization performance and 18.44% improvement in inductive learning performance.

## In ModernTSF
Default config: `configs/models/STOP.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/MaW0Z0025,
  author       = {Jiaming Ma and
                  Binwu Wang and
                  Pengkun Wang and
                  Zhengyang Zhou and
                  Xu Wang and
                  Yang Wang},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {Robust Spatio-Temporal Centralized Interaction for {OOD} Learning},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/ma25s.html},
  timestamp    = {Wed, 04 Feb 2026 17:22:46 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/MaW0Z0025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
