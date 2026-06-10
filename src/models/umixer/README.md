---
model: "UMixer"
forecasting_setting: "time_series"
config: "configs/models/UMixer.toml"
registry: "models.umixer.registry"
paper_title: "U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting"
venue: "AAAI 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2401.02236"
---
# UMixer

UMixer is a long-term time-series forecasting model published at AAAI 2024. It combines U-Net-style multi-scale skip connections with MLP-Mixer blocks to capture local temporal dependencies across patches and channels separately, and introduces a stationarity correction method that explicitly restores the non-stationary distribution of the data by constraining the difference in stationarity between the model input and output.

## Paper
- **Title**: U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting
- **Venue**: AAAI 2024
- **Published**: 2024 (arXiv: 2024-01)
- **arXiv**: https://arxiv.org/abs/2401.02236

## Abstract
Time series forecasting is a crucial task in various domains. Caused by factors such as trends, seasonality, or irregular fluctuations, time series often exhibits non-stationary. It obstructs stable feature propagation through deep layers, disrupts feature distributions, and complicates learning data distribution changes. As a result, many existing models struggle to capture the underlying patterns, leading to degraded forecasting performance. In this study, we tackle the challenge of non-stationarity in time series forecasting with our proposed framework called U-Mixer. By combining Unet and Mixer, U-Mixer effectively captures local temporal dependencies between different patches and channels separately to avoid the influence of distribution variations among channels, and merge low- and high-levels features to obtain comprehensive data representations. The key contribution is a novel stationarity correction method, explicitly restoring data distribution by constraining the difference in stationarity between the data before and after model processing to restore the non-stationarity information, while ensuring the temporal dependencies are preserved. Through extensive experiments on various real-world time series datasets, U-Mixer demonstrates its effectiveness and robustness, and achieves 14.5% and 7.7% improvements over state-of-the-art (SOTA) methods.

## In ModernTSF
Default config: `configs/models/UMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/Ma0FZ024,
  author       = {Xiang Ma and
                  Xuemei Li and
                  Lexin Fang and
                  Tianlong Zhao and
                  Caiming Zhang},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for
                  Time Series Forecasting},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {14255--14262},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i13.29337},
  doi          = {10.1609/AAAI.V38I13.29337},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/Ma0FZ024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
