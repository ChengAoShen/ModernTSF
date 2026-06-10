---
model: "PM25_GNN"
forecasting_setting: "covariate"
config: "configs/models/PM25_GNN.toml"
registry: "models.pm25gnn.registry"
paper_title: "PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting"
venue: "ACM SIGSPATIAL 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2002.12898"
---
# PM25_GNN

PM25_GNN is a graph neural network model for air quality (PM2.5 concentration) forecasting that integrates domain knowledge about pollutant diffusion processes to construct the graph topology and combines GNN layers with GRU-based temporal modeling to capture both fine-grained and long-term spatial-temporal dependencies across monitoring stations.

## Paper
- **Title**: PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting
- **Venue**: ACM SIGSPATIAL 2020
- **Published**: 2020 (arXiv: 2020-02)
- **arXiv**: https://arxiv.org/abs/2002.12898

## Abstract
When predicting PM2.5 concentrations, it is necessary to consider complex information sources since the concentrations are influenced by various factors within a long period. In this paper, we identify a set of critical domain knowledge for PM2.5 forecasting and develop a novel graph based model, PM2.5-GNN, being capable of capturing long-term dependencies. On a real-world dataset, we validate the effectiveness of the proposed model and examine its abilities of capturing both fine-grained and long-term influences in PM2.5 process. The proposed PM2.5-GNN has also been deployed online to provide free forecasting service.

## In ModernTSF
Default config: `configs/models/PM25_GNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/gis/WangLZMMG20,
  author       = {Shuo Wang and
                  Yanran Li and
                  Jiang Zhang and
                  Qingye Meng and
                  Lingwei Meng and
                  Fei Gao},
  editor       = {Chang{-}Tien Lu and
                  Fusheng Wang and
                  Goce Trajcevski and
                  Yan Huang and
                  Shawn D. Newsam and
                  Li Xiong},
  title        = {{PM2.5-GNN:} {A} Domain Knowledge Enhanced Graph Neural Network For
                  {PM2.5} Forecasting},
  booktitle    = {{SIGSPATIAL} '20: 28th International Conference on Advances in Geographic
                  Information Systems, Seattle, WA, USA, November 3-6, 2020},
  pages        = {163--166},
  publisher    = {{ACM}},
  year         = {2020},
  url          = {https://doi.org/10.1145/3397536.3422208},
  doi          = {10.1145/3397536.3422208},
  timestamp    = {Sat, 30 Sep 2023 09:41:50 +0200},
  biburl       = {https://dblp.org/rec/conf/gis/WangLZMMG20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
