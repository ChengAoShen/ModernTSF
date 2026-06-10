---
model: "CrossLinear"
forecasting_setting: "time_series"
config: "configs/models/CrossLinear.toml"
registry: "models.crosslinear.registry"
paper_title: "CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables"
venue: "KDD 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2505.23116"
---
# CrossLinear

CrossLinear is a linear-based time-series forecasting model designed for settings that include exogenous (external) variables. It incorporates a lightweight plug-and-play cross-correlation embedding module that captures time-invariant, direct variable dependencies between endogenous and exogenous channels while avoiding overfitting to time-varying or indirect dependencies. Patch-wise processing and a global linear head handle both short- and long-range temporal structure, serving the standard multivariate forecasting setting.

## Paper
- **Title**: CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables
- **Venue**: KDD 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.23116

## Abstract
Time series forecasting with exogenous variables is a critical emerging paradigm that presents unique challenges in modeling dependencies between variables. Traditional models often struggle to differentiate between endogenous and exogenous variables, leading to inefficiencies and overfitting. In this paper, we introduce CrossLinear, a novel Linear-based forecasting model that addresses these challenges by incorporating a plug-and-play cross-correlation embedding module. This lightweight module captures the dependencies between variables with minimal computational cost and seamlessly integrates into existing neural networks. Specifically, it captures time-invariant and direct variable dependencies while disregarding time-varying or indirect dependencies, thereby mitigating the risk of overfitting in dependency modeling and contributing to consistent performance improvements. Furthermore, CrossLinear employs patch-wise processing and a global linear head to effectively capture both short-term and long-term temporal dependencies, further improving its forecasting precision. Extensive experiments on 12 real-world datasets demonstrate that CrossLinear achieves superior performance in both short-term and long-term forecasting tasks. The ablation study underscores the effectiveness of the cross-correlation embedding module. Additionally, the generalizability of this module makes it a valuable plug-in for various forecasting tasks across different domains.

## In ModernTSF
Default config: `configs/models/CrossLinear.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/ZhouLL0025,
  author       = {Pengfei Zhou and
                  Yunlong Liu and
                  Junli Liang and
                  Qi Song and
                  Xiangyang Li},
  editor       = {Luiza Antonie and
                  Jian Pei and
                  Xiaohui Yu and
                  Flavio Chierichetti and
                  Hady W. Lauw and
                  Yizhou Sun and
                  Srinivasan Parthasarathy},
  title        = {CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series
                  Forecasting with Exogenous Variables},
  booktitle    = {Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, V.2, {KDD} 2025, Toronto ON, Canada, August 3-7,
                  2025},
  pages        = {4120--4131},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3711896.3736899},
  doi          = {10.1145/3711896.3736899},
  timestamp    = {Wed, 24 Dec 2025 10:44:06 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/ZhouLL0025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
