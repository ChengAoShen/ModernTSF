---
model: "FITS"
forecasting_setting: "time_series"
config: "configs/models/FITS.toml"
registry: "models.fits.registry"
paper_title: "FITS: Modeling Time Series with 10k Parameters"
venue: "ICLR 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2307.03756"
---
# FITS

FITS (Frequency Interpolation Time Series analysis) is a lightweight time series forecasting model that operates entirely in the complex frequency domain. Instead of processing raw time-domain sequences, FITS applies rFFT to compress the input, performs low-pass filtering to discard high-frequency noise, and uses frequency-domain interpolation to map the compressed representation to the target prediction length, enabling competitive forecasting performance with only approximately 10k parameters — small enough for edge-device deployment.

## Paper
- **Title**: FITS: Modeling Time Series with 10k Parameters
- **Venue**: ICLR 2024 (Spotlight)
- **Published**: 2024 (arXiv: 2023-07)
- **arXiv**: https://arxiv.org/abs/2307.03756

## Abstract
In this paper, we introduce FITS, a lightweight yet powerful model for time series analysis. Unlike existing models that directly process raw time-domain data, FITS operates on the principle that time series can be manipulated through interpolation in the complex frequency domain. By discarding high-frequency components with negligible impact on time series data, FITS achieves performance comparable to state-of-the-art models for time series forecasting and anomaly detection tasks, while having a remarkably compact size of only approximately $10k$ parameters. Such a lightweight model can be easily trained and deployed in edge devices, creating opportunities for various applications.

## In ModernTSF
Default config: `configs/models/FITS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/XuZ024,
  author       = {Zhijian Xu and
                  Ailing Zeng and
                  Qiang Xu},
  title        = {{FITS:} Modeling Time Series with 10k Parameters},
  booktitle    = {The Twelfth International Conference on Learning Representations,
                  {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=bWcnvZ3qMb},
  timestamp    = {Mon, 29 Jul 2024 17:17:48 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/XuZ024.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
