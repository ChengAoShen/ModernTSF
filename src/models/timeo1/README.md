---
model: "TimeO1"
forecasting_setting: "time_series"
config: "configs/models/TimeO1.toml"
registry: "models.timeo1.registry"
paper_title: "Time-o1: Time-Series Forecasting Needs Transformed Label Alignment"
venue: "NeurIPS 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2505.17847"
---
# TimeO1

TimeO1 is a time series forecasting approach that improves training through a transformation-augmented learning objective: it transforms the label sequence into decorrelated components ranked by significance, then trains the model to align only the most important components, addressing both label autocorrelation bias and the excessive task complexity that grows with the forecast horizon under standard mean squared error training.

## Paper
- **Title**: Time-o1: Time-Series Forecasting Needs Transformed Label Alignment
- **Venue**: NeurIPS 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.17847

## Abstract
Training time-series forecast models presents unique challenges in designing effective learning objectives. Existing methods predominantly utilize the temporal mean squared error, which faces two critical challenges: (1) label autocorrelation, which leads to bias from the label sequence likelihood; (2) excessive amount of tasks, which increases with the forecast horizon and complicates optimization. To address these challenges, we propose Time-o1, a transformation-augmented learning objective tailored for time-series forecasting. The central idea is to transform the label sequence into decorrelated components with discriminated significance. Models are then trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Extensive experiments demonstrate that Time-o1 achieves state-of-the-art performance and is compatible with various forecast models. Code is available at https://github.com/Master-PLC/Time-o1.

## In ModernTSF
Default config: `configs/models/TimeO1.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{wang2025timeo,
  author        = {Hao Wang and
                  Licheng Pan and
                  Zhichao Chen and
                  Xu Chen and
                  Qingyang Dai and
                  Lei Wang and
                  Haoxuan Li and
                  Zhouchen Lin},
  title         = {Time-o1: Time-Series Forecasting Needs Transformed Label Alignment},
  year          = {2025},
  eprint        = {2505.17847},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2505.17847}
}
```
