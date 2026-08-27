---
name: "GCLSTM"
implementation: rewrite
summary: "GCLSTM (Graph Convolutional LSTM) is a covariate prediction model for node-level air-quality forecasting on graph-structured sensor networks. This repository implementation applies Chebyshev spectral graph convolution to each historical step and then processes the resulting sequence with a custom LSTM. It consumes historical values and calendar covariates and predicts future concentrations at all nodes."
paper:
  title: "A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory"
  venue: "Science of the Total Environment 2019"
  year: 2019
  url: "https://doi.org/10.1016/j.scitotenv.2019.01.333"
codebase:
  url: "https://github.com/PoorOtterBob/CauAir"
  revision: "73dae00ca6ad14abb15174a0a0286d500e868b94"
  license: "NOASSERTION"
  usage: reference-only
---
# GCLSTM

GCLSTM (Graph Convolutional LSTM) is a covariate prediction model for node-level air-quality forecasting on graph-structured sensor networks. This repository implementation applies Chebyshev spectral graph convolution to each historical step and then processes the resulting sequence with a custom LSTM. It consumes historical values and calendar covariates and predicts future concentrations at all nodes.

## Paper
- **Title**: A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory
- **Venue**: Science of the Total Environment, vol. 664, pp. 1-10
- **Published**: 2019
- **arXiv**: N/A

## Abstract
In this paper, we developed a hybrid deep learning approach, which integrates Graph Convolutional networks and Long Short-Term Memory networks (GC-LSTM), to model and forecast the spatiotemporal variation of PM2.5 concentrations. We model historical observations on different stations as spatiotemporal graph series, where air quality variables, meteorological factors, and temporal attributes were used as graph signals. Graph convolutional networks (GCN) were applied to extract the spatial dependency between different stations and LSTM to capture the temporal dependency among observations at different times. The GC-LSTM was trained and tested on real-world data and compared with other state-of-the-art methods. The results showed that GC-LSTM achieved the best performance for predictions with a recall rate of 68.45%, false alarm rate of 4.65% (both at threshold: 115 μg/m³) and correlation coefficient R² of 0.72 for 72-hour forecasts. In addition to PM2.5, the proposed methodology could also be applied to concentration forecasting of different air pollutants in future.

## In ModernTSF
Default config: `configs/models/GCLSTM.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Implementation: **rewrite** (clean-room audit pending). No author-released code was identified. The
immediate implementation source is
[`PoorOtterBob/CauAir`](https://github.com/PoorOtterBob/CauAir) at revision
`73dae00ca6ad14abb15174a0a0286d500e868b94`, whose repository declares no
license. The paper-level graph-convolution/LSTM idea is present, but the exact
feature pipeline, graph construction, optimization protocol, and numerical
parity cannot be established from first-party evidence. The local direct
multi-horizon decoder and shared runner objective are explicit deviations.

## Citation

```bibtex
@article{Qi2019GCLSTM,
  author    = {Yanlin Qi and
               Qi Li and
               Hamed Karimian and
               Di Liu},
  title     = {A hybrid model for spatiotemporal forecasting of PM2.5 based on graph
               convolutional neural network and long short-term memory},
  journal   = {Science of The Total Environment},
  volume    = {664},
  pages     = {1--10},
  year      = {2019},
  doi       = {10.1016/J.SCITOTENV.2019.01.333}
}
```
