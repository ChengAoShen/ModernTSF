---
model: "StemGNN"
forecasting_setting: "spatiotemporal"
config: "configs/models/StemGNN.toml"
registry: "models.stemgnn.registry"
paper_title: "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting"
venue: "NeurIPS 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2103.07719"
---
# StemGNN

StemGNN (Spectral Temporal Graph Neural Network) is a spatiotemporal model for multivariate time-series forecasting that captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines a Graph Fourier Transform (GFT) for spatial correlation and a Discrete Fourier Transform (DFT) for temporal patterns in a unified end-to-end framework, learning the inter-series graph structure automatically from data without pre-defined priors.

## Paper
- **Title**: Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting
- **Venue**: NeurIPS 2020
- **Published**: 2020 (arXiv: 2021-03)
- **arXiv**: https://arxiv.org/abs/2103.07719

## Abstract
Multivariate time-series forecasting plays a crucial role in many real-world applications. It is a challenging problem as one needs to consider both intra-series temporal correlations and inter-series correlations simultaneously. Recently, there have been multiple works trying to capture both correlations, but most, if not all of them only capture temporal correlations in the time domain and resort to pre-defined priors as inter-series relationships. In this paper, we propose Spectral Temporal Graph Neural Network (StemGNN) to further improve the accuracy of multivariate time-series forecasting. StemGNN captures inter-series correlations and temporal dependencies jointly in the spectral domain. It combines Graph Fourier Transform (GFT) which models inter-series correlations and Discrete Fourier Transform (DFT) which models temporal dependencies in an end-to-end framework. After passing through GFT and DFT, the spectral representations hold clear patterns and can be predicted effectively by convolution and sequential learning modules. Moreover, StemGNN learns inter-series correlations automatically from the data without using pre-defined priors. We conduct extensive experiments on ten real-world datasets to demonstrate the effectiveness of StemGNN. Code is available at https://github.com/microsoft/StemGNN/

## In ModernTSF
Default config: `configs/models/StemGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/CaoWDZZHTXBTZ20,
  author       = {Defu Cao and
                  Yujing Wang and
                  Juanyong Duan and
                  Ce Zhang and
                  Xia Zhu and
                  Congrui Huang and
                  Yunhai Tong and
                  Bixiong Xu and
                  Jing Bai and
                  Jie Tong and
                  Qi Zhang},
  editor       = {Hugo Larochelle and
                  Marc'Aurelio Ranzato and
                  Raia Hadsell and
                  Maria{-}Florina Balcan and
                  Hsuan{-}Tien Lin},
  title        = {Spectral Temporal Graph Neural Network for Multivariate Time-series
                  Forecasting},
  booktitle    = {Advances in Neural Information Processing Systems 33: Annual Conference
                  on Neural Information Processing Systems 2020, NeurIPS 2020, December
                  6-12, 2020, virtual},
  year         = {2020},
  url          = {https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html},
  timestamp    = {Fri, 16 Jan 2026 08:42:14 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/CaoWDZZHTXBTZ20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
