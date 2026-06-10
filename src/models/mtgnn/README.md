---
model: "MTGNN"
forecasting_setting: "spatiotemporal"
config: "configs/models/MTGNN.toml"
registry: "models.mtgnn.registry"
paper_title: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"
venue: "KDD 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2005.11650"
---
# MTGNN

MTGNN is a spatiotemporal graph neural network for multivariate time-series forecasting that jointly learns the graph structure and performs message passing. It uses a graph learning module to automatically extract uni-directed inter-variable relations, a mix-hop propagation layer for multi-hop spatial aggregation, and dilated inception layers for multi-scale temporal convolution, all trained end-to-end without requiring a pre-defined graph.

## Paper
- **Title**: Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks
- **Venue**: KDD 2020
- **Published**: 2020 (arXiv: 2020-05)
- **arXiv**: https://arxiv.org/abs/2005.11650

## Abstract
Modeling multivariate time series has long been a subject that has attracted researchers from a diverse range of fields including economics, finance, and traffic. A basic assumption behind multivariate time series forecasting is that its variables depend on one another but, upon looking closely, it is fair to say that existing methods fail to fully exploit latent spatial dependencies between pairs of variables. In recent years, meanwhile, graph neural networks (GNNs) have shown high capability in handling relational dependencies. GNNs require well-defined graph structures for information propagation which means they cannot be applied directly for multivariate time series where the dependencies are not known in advance. In this paper, we propose a general graph neural network framework designed specifically for multivariate time series data. Our approach automatically extracts the uni-directed relations among variables through a graph learning module, into which external knowledge like variable attributes can be easily integrated. A novel mix-hop propagation layer and a dilated inception layer are further proposed to capture the spatial and temporal dependencies within the time series. The graph learning, graph convolution, and temporal convolution modules are jointly learned in an end-to-end framework. Experimental results show that our proposed model outperforms the state-of-the-art baseline methods on 3 of 4 benchmark datasets and achieves on-par performance with other approaches on two traffic datasets which provide extra structural information.

## In ModernTSF
Default config: `configs/models/MTGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/WuPL0CZ20,
  author       = {Zonghan Wu and
                  Shirui Pan and
                  Guodong Long and
                  Jing Jiang and
                  Xiaojun Chang and
                  Chengqi Zhang},
  editor       = {Rajesh Gupta and
                  Yan Liu and
                  Jiliang Tang and
                  B. Aditya Prakash},
  title        = {Connecting the Dots: Multivariate Time Series Forecasting with Graph
                  Neural Networks},
  booktitle    = {{KDD} '20: The 26th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, Virtual Event, CA, USA, August 23-27, 2020},
  pages        = {753--763},
  publisher    = {{ACM}},
  year         = {2020},
  url          = {https://doi.org/10.1145/3394486.3403118},
  doi          = {10.1145/3394486.3403118},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/WuPL0CZ20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
