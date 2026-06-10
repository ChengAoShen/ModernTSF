---
model: "CrossGNN"
forecasting_setting: "time_series"
config: "configs/models/CrossGNN.toml"
registry: "models.crossgnn.registry"
paper_title: "CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement"
venue: "NeurIPS 2023"
year: 2023
arxiv: ""
---
# CrossGNN

CrossGNN is a multivariate time-series forecasting model that tackles noise and inter-variable heterogeneity through a linear-complexity graph neural network framework. It uses an adaptive multi-scale identifier to build cleaner multi-resolution views of the input, a Cross-Scale GNN to capture trend information at the most informative scale, and a Cross-Variable GNN to jointly model homogeneity and heterogeneity between channels — all while maintaining O(L) time and space complexity with respect to sequence length.

## Paper
- **Title**: CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement
- **Venue**: NeurIPS 2023
- **Published**: 2023
- **arXiv**: N/A

## Abstract
Recently, multivariate time series (MTS) forecasting techniques have seen rapid development and widespread applications across various fields. Transformer-based and GNN-based methods have shown promising potential due to their strong ability to model interaction of time and variables. However, by conducting a comprehensive analysis of the real-world data, we observe that the temporal fluctuations and heterogeneity between variables are not well handled by existing methods. To address the above issues, we propose CrossGNN, a linear complexity GNN model to refine the cross-scale and cross-variable interaction for MTS. To deal with the unexpected noise in time dimension, an adaptive multi-scale identifier (AMSI) is leveraged to construct multi-scale time series with reduced noise. A Cross-Scale GNN is proposed to extract the scales with clearer trend and weaker noise. Cross-Variable GNN is proposed to utilize the homogeneity and heterogeneity between different variables. By simultaneously focusing on edges with higher saliency scores and constraining those edges with lower scores, the time and space complexity (i.e., O(L)) of CrossGNN can be linear with the input sequence length L. Extensive experimental results on 8 real-world MTS datasets demonstrate the effectiveness of CrossGNN compared with state-of-the-art methods.

## In ModernTSF
Default config: `configs/models/CrossGNN.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/HuangSZDWZW23,
  author       = {Qihe Huang and
                  Lei Shen and
                  Ruixin Zhang and
                  Shouhong Ding and
                  Binwu Wang and
                  Zhengyang Zhou and
                  Yang Wang},
  editor       = {Alice Oh and
                  Tristan Naumann and
                  Amir Globerson and
                  Kate Saenko and
                  Moritz Hardt and
                  Sergey Levine},
  title        = {CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction
                  Refinement},
  booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference
                  on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
                  LA, USA, December 10 - 16, 2023},
  year         = {2023},
  url          = {http://papers.nips.cc/paper\_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html},
  timestamp    = {Tue, 26 Mar 2024 15:54:05 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/HuangSZDWZW23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
