---
model: "ModernTCN"
forecasting_setting: "time_series"
config: "configs/models/ModernTCN.toml"
registry: "models.moderntcn.registry"
paper_title: "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"
venue: "ICLR 2024"
year: 2024
arxiv: ""
---
# ModernTCN

ModernTCN is a pure convolutional architecture for general time series analysis that modernizes the traditional Temporal Convolutional Network (TCN) by incorporating large effective receptive fields through depthwise separable convolutions, achieving state-of-the-art performance across long-term and short-term forecasting, imputation, classification, and anomaly detection tasks.

## Paper
- **Title**: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
- **Venue**: ICLR 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Recently, Transformer-based and MLP-based models have emerged rapidly and won dominance in time series analysis. In contrast, convolution is losing steam in time series tasks nowadays for inferior performance. This paper studies the open question of how to better use convolution in time series analysis and makes efforts to bring convolution back to the arena of time series analysis. To this end, we modernize the traditional TCN and conduct time series related modifications to make it more suitable for time series tasks. As the outcome, we propose ModernTCN and successfully solve this open question through a seldom-explored way in time series community. As a pure convolution structure, ModernTCN still achieves the consistent state-of-the-art performance on five mainstream time series analysis tasks while maintaining the efficiency advantage of convolution-based models, therefore providing a better balance of efficiency and performance than state-of-the-art Transformer-based and MLP-based models. Our study further reveals that, compared with previous convolution-based models, our ModernTCN has much larger effective receptive fields (ERFs), therefore can better unleash the potential of convolution in time series analysis.

## In ModernTSF
Default config: `configs/models/ModernTCN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LuoW24,
  author       = {Donghao Luo and
                  Xue Wang},
  title        = {ModernTCN: {A} Modern Pure Convolution Structure for General Time
                  Series Analysis},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=vpJMJerXHU},
  code         = {https://github.com/luodhhh/ModernTCN},
  timestamp    = {Thu, 22 May 2025 17:54:02 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/LuoW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
