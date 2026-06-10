---
model: "DTAF"
forecasting_setting: "time_series"
config: "configs/models/DTAF.toml"
registry: "models.dtaf.registry"
paper_title: "Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2511.08229"
---
# DTAF

DTAF is a dual-branch time series forecasting framework designed to handle non-stationary data by simultaneously addressing temporal distribution shifts and spectral variability: the Temporal Stabilizing Fusion (TFS) module suppresses non-stationary temporal patterns via a mixture-of-experts filter while the Frequency Wave Modeling (FWM) module applies frequency differencing to highlight spectral shifts, with the two branches fused for robust long-term predictions.

## Paper
- **Title**: Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-11)
- **arXiv**: https://arxiv.org/abs/2511.08229

## Abstract
Time series forecasting is critical for decision-making across dynamic domains such as energy, finance, transportation, and cloud computing. However, real-world time series often exhibit non-stationarity, including temporal distribution shifts and spectral variability, which pose significant challenges for long-term time series forecasting. In this paper, we propose DTAF, a dual-branch framework that addresses non-stationarity in both the temporal and frequency domains. For the temporal domain, the Temporal Stabilizing Fusion (TFS) module employs a non-stationary mix of experts (MOE) filter to disentangle and suppress temporal non-stationary patterns while preserving long-term dependencies. For the frequency domain, the Frequency Wave Modeling (FWM) module applies frequency differencing to dynamically highlight components with significant spectral shifts. By fusing the complementary outputs of TFS and FWM, DTAF generates robust forecasts that adapt to both temporal and frequency domain non-stationarity. Extensive experiments on real-world benchmarks demonstrate that DTAF outperforms state-of-the-art baselines, yielding significant improvements in forecasting accuracy under non-stationary conditions. All codes are available at https://github.com/decisionintelligence/DTAF.

## In ModernTSF
Default config: `configs/models/DTAF.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LuCGSWY26,
  author       = {Junkai Lu and
                  Peng Chen and
                  Chenjuan Guo and
                  Yang Shu and
                  Meng Wang and
                  Bin Yang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Towards Non-Stationary Time Series Forecasting with Temporal Stabilization
                  and Frequency Differencing},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {24070--24078},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i29.39585},
  doi          = {10.1609/AAAI.V40I29.39585},
  timestamp    = {Wed, 25 Mar 2026 16:59:58 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/LuCGSWY26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
