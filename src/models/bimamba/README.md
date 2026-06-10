---
model: "BiMamba"
forecasting_setting: "time_series"
config: "configs/models/BiMamba.toml"
registry: "models.bimamba.registry"
paper_title: "Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting"
venue: "arXiv preprint"
year: 2024
arxiv: "https://arxiv.org/abs/2404.15772"
---
# BiMamba

BiMamba is a bidirectional state-space model (SSM) for long-term multivariate time-series forecasting. It extends the Mamba selective SSM with a forget gate (Mamba+) and runs it in both the forward and backward directions, enabling the model to capture long-range temporal dependencies without the quadratic cost of Transformer attention. A series-relation-aware decider automatically selects between channel-independent and channel-mixing tokenisation strategies depending on the dataset.

## Paper
- **Title**: Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2024 (arXiv: 2024-04)
- **arXiv**: https://arxiv.org/abs/2404.15772

## Abstract
Long-term time series forecasting (LTSF) provides longer insights into future trends and patterns. Over the past few years, deep learning models especially Transformers have achieved advanced performance in LTSF tasks. However, LTSF faces inherent challenges such as long-term dependencies capturing and sparse semantic characteristics. Recently, a new state space model (SSM) named Mamba is proposed. With the selective capability on input data and the hardware-aware parallel computing algorithm, Mamba has shown great potential in balancing predicting performance and computational efficiency compared to Transformers. To enhance Mamba's ability to preserve historical information in a longer range, we design a novel Mamba+ block by adding a forget gate inside Mamba to selectively combine the new features with the historical features in a complementary manner. Furthermore, we apply Mamba+ both forward and backward and propose Bi-Mamba+, aiming to promote the model's ability to capture interactions among time series elements. Additionally, multivariate time series data in different scenarios may exhibit varying emphasis on intra- or inter-series dependencies. Therefore, we propose a series-relation-aware decider that controls the utilization of channel-independent or channel-mixing tokenization strategy for specific datasets. Extensive experiments on 8 real-world datasets show that our model achieves better predictions compared with state-of-the-art methods. Our code is available at https://github.com/Leopold2333/Bi-Mamba+.

## In ModernTSF
Default config: `configs/models/BiMamba.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{liang2024bimamba,
  author        = {Aobo Liang and
                  Xingguo Jiang and
                  Yan Sun and
                  Xiaohou Shi and
                  Ke Li},
  title         = {Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting},
  year          = {2024},
  eprint        = {2404.15772},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2404.15772}
}
```
