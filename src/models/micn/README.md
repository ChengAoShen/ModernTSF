---
model: "MICN"
forecasting_setting: "time_series"
config: "configs/models/MICN.toml"
registry: "models.micn.registry"
paper_title: "MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting"
venue: "ICLR 2023"
year: 2023
arxiv: ""
---
# MICN

MICN (Multi-scale Isometric Convolution Network) is a long-term time-series forecasting model presented at ICLR 2023. It adopts a multi-scale branch structure where each branch extracts local temporal features via down-sampled convolution and captures global correlations via isometric convolution, achieving linear complexity with respect to sequence length while outperforming Transformer-based methods on standard benchmarks.

## Paper
- **Title**: MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting
- **Venue**: ICLR 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract
Recently, Transformer-based methods have achieved surprising performance in the field of long-term series forecasting, but the attention mechanism for computing global correlations entails high complexity. And they do not allow for targeted modeling of local features as CNN structures do. To solve the above problems, we propose to combine local features and global correlations to capture the overall view of time series (e.g., fluctuations, trends). To fully exploit the underlying information in the time series, a multi-scale branch structure is adopted to model different potential patterns separately. Each pattern is extracted with down-sampled convolution and isometric convolution for local features and global correlations, respectively. In addition to being more effective, our proposed method, termed as Multi-scale Isometric Convolution Network (MICN), is more efficient with linear complexity about the sequence length with suitable convolution kernels. Our experiments on six benchmark datasets show that compared with state-of-the-art methods, MICN yields 17.2% and 21.6% relative improvements for multivariate and univariate time series, respectively.

## In ModernTSF
Default config: `configs/models/MICN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/Wang0HWCX23,
  author       = {Huiqiang Wang and
                  Jian Peng and
                  Feihu Huang and
                  Jince Wang and
                  Junhui Chen and
                  Yifei Xiao},
  title        = {{MICN:} Multi-scale Local and Global Context Modeling for Long-term
                  Series Forecasting},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/forum?id=zt53IDUR1U},
  timestamp    = {Mon, 21 Oct 2024 15:07:23 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/Wang0HWCX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
