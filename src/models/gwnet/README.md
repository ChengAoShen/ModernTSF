---
model: "GWNet"
forecasting_setting: "spatiotemporal"
config: "configs/models/GWNet.toml"
registry: "models.gwnet.registry"
paper_title: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
venue: "IJCAI 2019"
year: 2019
arxiv: "https://arxiv.org/abs/1906.00121"
---
# GWNet

GWNet (Graph WaveNet) is a spatiotemporal graph neural network that serves the spatiotemporal forecasting setting on node-structured data. It jointly models hidden spatial dependencies via a learned adaptive adjacency matrix and long-range temporal trends via stacked dilated 1D causal convolutions whose receptive field grows exponentially with depth — enabling end-to-end, scalable traffic and sensor-network forecasting.

## Paper
- **Title**: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
- **Venue**: IJCAI 2019
- **Published**: 2019 (arXiv: 2019-05)
- **arXiv**: https://arxiv.org/abs/1906.00121

## Abstract
Spatial-temporal graph modeling is an important task to analyze the spatial relations and temporal trends of components in a system. Existing approaches mostly capture the spatial dependency on a fixed graph structure, assuming that the underlying relation between entities is pre-determined. However, the explicit graph structure (relation) does not necessarily reflect the true dependency and genuine relation may be missing due to the incomplete connections in the data. Furthermore, existing methods are ineffective to capture the temporal trends as the RNNs or CNNs employed in these methods cannot capture long-range temporal sequences. To overcome these limitations, we propose in this paper a novel graph neural network architecture, Graph WaveNet, for spatial-temporal graph modeling. By developing a novel adaptive dependency matrix and learn it through node embedding, our model can precisely capture the hidden spatial dependency in the data. With a stacked dilated 1D convolution component whose receptive field grows exponentially as the number of layers increases, Graph WaveNet is able to handle very long sequences. These two components are integrated seamlessly in a unified framework and the whole framework is learned in an end-to-end manner. Experimental results on two public traffic network datasets, METR-LA and PEMS-BAY, demonstrate the superior performance of our algorithm.

## In ModernTSF
Default config: `configs/models/GWNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/ijcai/WuPLJZ19,
  author       = {Zonghan Wu and
                  Shirui Pan and
                  Guodong Long and
                  Jing Jiang and
                  Chengqi Zhang},
  editor       = {Sarit Kraus},
  title        = {Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  booktitle    = {Proceedings of the Twenty-Eighth International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2019, Macao, China, August 10-16,
                  2019},
  pages        = {1907--1913},
  publisher    = {ijcai.org},
  year         = {2019},
  url          = {https://doi.org/10.24963/ijcai.2019/264},
  doi          = {10.24963/IJCAI.2019/264},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/ijcai/WuPLJZ19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
