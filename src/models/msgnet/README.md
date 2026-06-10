---
model: "MSGNet"
forecasting_setting: "time_series"
config: "configs/models/MSGNet.toml"
registry: "models.msgnet.registry"
paper_title: "MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting"
venue: "AAAI 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2401.00423"
---
# MSGNet

MSGNet is a time series forecasting model for multivariate sequence prediction. It captures varying inter-series correlations across multiple time scales by combining frequency domain analysis (FFT-based period extraction) with an adaptive mixhop graph convolution layer, while self-attention handles intra-series dependencies within each scale — all without requiring an external adjacency matrix.

## Paper
- **Title**: MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2401.00423

## Abstract
Multivariate time series forecasting poses an ongoing challenge across various disciplines. Time series data often exhibit diverse intra-series and inter-series correlations, contributing to intricate and interwoven dependencies that have been the focus of numerous studies. Nevertheless, a significant research gap remains in comprehending the varying inter-series correlations across different time scales among multiple time series, an area that has received limited attention in the literature. To bridge this gap, this paper introduces MSGNet, an advanced deep learning model designed to capture the varying inter-series correlations across multiple time scales using frequency domain analysis and adaptive graph convolution. By leveraging frequency domain analysis, MSGNet effectively extracts salient periodic patterns and decomposes the time series into distinct time scales. The model incorporates a self-attention mechanism to capture intra-series dependencies, while introducing an adaptive mixhop graph convolution layer to autonomously learn diverse inter-series correlations within each time scale. Extensive experiments are conducted on several real-world datasets to showcase the effectiveness of MSGNet. Furthermore, MSGNet possesses the ability to automatically learn explainable multi-scale inter-series correlations, exhibiting strong generalization capabilities even when applied to out-of-distribution samples.

## In ModernTSF
Default config: `configs/models/MSGNet.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/CaiLLFW24,
  author       = {Wanlin Cai and
                  Yuxuan Liang and
                  Xianggen Liu and
                  Jianshuai Feng and
                  Yuankai Wu},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {MSGNet: Learning Multi-Scale Inter-series Correlations for Multivariate
                  Time Series Forecasting},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {11141--11149},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i10.28991},
  doi          = {10.1609/AAAI.V38I10.28991},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/CaiLLFW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
