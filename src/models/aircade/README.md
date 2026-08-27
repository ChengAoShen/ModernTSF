---
model: "AirCade"
forecasting_setting: "covariate"
config: "configs/models/AirCade.toml"
spec: "models.aircade.spec"
paper_title: "Spatiotemporal Causal Decoupling Model for Air Quality Forecasting"
venue: "ICASSP 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2505.20119"
---
# AirCade

AirCade is a spatiotemporal causal-decoupling model for air quality index (AQI) forecasting that serves the covariate prediction setting. It uses a spatiotemporal Transformer with knowledge-embedding techniques to capture internal AQI dynamics, disentangles synchronous causality between past AQI and meteorological features via a causal decoupling module, and introduces a causal intervention mechanism to represent uncertainty in future meteorological features — enabling robust, future-covariate-aware node-level predictions.

## Paper
- **Title**: Spatiotemporal Causal Decoupling Model for Air Quality Forecasting
- **Venue**: ICASSP 2025
- **Published**: 2025 (arXiv: 2025-05)
- **arXiv**: https://arxiv.org/abs/2505.20119

## Abstract
Due to the profound impact of air pollution on human health, livelihoods, and economic development, air quality forecasting is of paramount significance. Initially, we employ the causal graph method to scrutinize the constraints of existing research in comprehensively modeling the causal relationships between the air quality index (AQI) and meteorological features. In order to enhance prediction accuracy, we introduce a novel air quality forecasting model, AirCade, which incorporates a causal decoupling approach. AirCade leverages a spatiotemporal module in conjunction with knowledge embedding techniques to capture the internal dynamics of AQI. Subsequently, a causal decoupling module is proposed to disentangle synchronous causality from past AQI and meteorological features, followed by the dissemination of acquired knowledge to future time steps to enhance performance. Additionally, we introduce a causal intervention mechanism to explicitly represent the uncertainty of future meteorological features, thereby bolstering the model's robustness. Our evaluation of AirCade on an open-source air quality dataset demonstrates over 20% relative improvement over state-of-the-art models. Our source code is available at https://github.com/PoorOtterBob/AirCade.

## In ModernTSF
Default config: `configs/models/AirCade.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Official source: https://github.com/PoorOtterBob/AirCade at `179067f5b9fbc05f894022809e0b1c83e9f61fd8` (no license file declared at that revision).
- Evidence: `unverified`. The core model file matches the pinned source apart from its `BaseModel` import, but numerical parity and the original experiment pipeline have not been established.
- Known differences: node embeddings are resized from the fixed 184-station layout, `pred_len` must equal `seq_len`, generic time marks can stand in for future meteorological covariates, and the preset is smaller than the official constructor defaults. The official frequency-domain training objective is not reproduced by the generic loop.

## Citation

```bibtex
@inproceedings{DBLP:conf/icassp/MaWHYWWW25,
  author       = {Jiaming Ma and
                  Guanjun Wang and
                  Sheng Huang and
                  Kuo Yang and
                  Binwu Wang and
                  Pengkun Wang and
                  Yang Wang},
  title        = {Spatiotemporal Causal Decoupling Model for Air Quality Forecasting},
  booktitle    = {2025 {IEEE} International Conference on Acoustics, Speech and Signal
                  Processing, {ICASSP} 2025, Hyderabad, India, April 6-11, 2025},
  pages        = {1--5},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/ICASSP49660.2025.11099015},
  doi          = {10.1109/ICASSP49660.2025.11099015},
  timestamp    = {Wed, 11 Feb 2026 11:45:24 +0100},
  biburl       = {https://dblp.org/rec/conf/icassp/MaWHYWWW25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
