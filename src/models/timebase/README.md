---
model: "TimeBase"
forecasting_setting: "time_series"
config: "configs/models/TimeBase.toml"
registry: "models.timebase.registry"
paper_title: "TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting"
venue: "ICML 2025"
year: 2025
arxiv: ""
---
# TimeBase

TimeBase is an ultra-lightweight network for long-term time series forecasting that extracts core basis temporal components from the input window and transforms traditional point-level prediction into efficient segment-level forecasting, exploiting the temporal pattern similarity and low-rank structure inherent in long-horizon time series data.

## Paper
- **Title**: TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting
- **Venue**: ICML 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
Long-term time series forecasting (LTSF) has traditionally relied on large parameters to capture extended temporal dependencies, resulting in substantial computational costs and inefficiencies in both memory usage and processing time. However, time series data, unlike high-dimensional images or text, often exhibit temporal pattern similarity and low-rank structures, especially in long-term horizons. By leveraging this structure, models can be guided to focus on more essential, concise temporal data, improving both accuracy and computational efficiency. In this paper, we introduce TimeBase, an ultra-lightweight network to harness the power of minimalism in LTSF. TimeBase 1) extracts core basis temporal components and 2) transforms traditional point-level forecasting into efficient segment-level forecasting, achieving optimal utilization of both data and parameters. Extensive experiments on diverse real-world datasets show that TimeBase achieves remarkable efficiency and secures competitive forecasting performance. Additionally, TimeBase can also serve as a very effective plug-and-play complexity reducer for any patch-based forecasting models.

## In ModernTSF
Default config: `configs/models/TimeBase.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/HuangZYYW025,
  author       = {Qihe Huang and
                  Zhengyang Zhou and
                  Kuo Yang and
                  Zhongchao Yi and
                  Xu Wang and
                  Yang Wang},
  editor       = {Aarti Singh and
                  Maryam Fazel and
                  Daniel Hsu and
                  Simon Lacoste{-}Julien and
                  Felix Berkenkamp and
                  Tegan Maharaj and
                  Kiri Wagstaff and
                  Jerry Zhu},
  title        = {TimeBase: The Power of Minimalism in Efficient Long-term Time Series
                  Forecasting},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML}
                  2025, Vancouver, BC, Canada, July 13-19, 2025},
  series       = {Proceedings of Machine Learning Research},
  publisher    = {{PMLR} / OpenReview.net},
  year         = {2025},
  url          = {https://proceedings.mlr.press/v267/huang25az.html},
  timestamp    = {Thu, 12 Feb 2026 07:51:25 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/HuangZYYW025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
