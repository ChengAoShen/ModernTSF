---
model: "GCLSTM"
forecasting_setting: "covariate"
config: "configs/models/GCLSTM.toml"
registry: "models.gclstm.registry"
paper_title: "A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory"
venue: "Science of the Total Environment 2019"
year: 2019
arxiv: ""
---
# GCLSTM

GCLSTM (Graph Convolutional LSTM) is a covariate prediction model for node-level air-quality forecasting on graph-structured sensor networks. It integrates Chebyshev spectral graph convolution directly inside an LSTM cell so that at every recurrent time step the hidden state is updated by propagating information across adjacent sensor nodes before the gating computation. The model consumes historical pollutant values together with meteorological covariates and predicts future pollutant concentrations at all nodes, operating in the covariate spatiotemporal forecasting setting.

## Paper
- **Title**: A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory
- **Venue**: Science of the Total Environment, vol. 664, pp. 1-10
- **Published**: 2019
- **arXiv**: N/A

## Abstract
In this paper, we developed a hybrid deep learning approach, which integrates Graph Convolutional networks and Long Short-Term Memory networks (GC-LSTM), to model and forecast the spatiotemporal variation of PM2.5 concentrations. We model historical observations on different stations as spatiotemporal graph series, where air quality variables, meteorological factors, and temporal attributes were used as graph signals. Graph convolutional networks (GCN) were applied to extract the spatial dependency between different stations and LSTM to capture the temporal dependency among observations at different times. The GC-LSTM was trained and tested on real-world data and compared with other state-of-the-art methods. The results showed that GC-LSTM achieved the best performance for predictions with a recall rate of 68.45%, false alarm rate of 4.65% (both at threshold: 115 μg/m³) and correlation coefficient R² of 0.72 for 72-hour forecasts. In addition to PM2.5, the proposed methodology could also be applied to concentration forecasting of different air pollutants in future.

## In ModernTSF
Default config: `configs/models/GCLSTM.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
