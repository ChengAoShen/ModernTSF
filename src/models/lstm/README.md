---
model: "LSTM"
forecasting_setting: "spatiotemporal"
config: "configs/models/LSTM.toml"
registry: "models.lstm.registry"
paper_title: "Long Short-Term Memory"
venue: "Neural Computation 1997"
year: 1997
arxiv: ""
---
# LSTM

LSTM is a per-node vanilla Long Short-Term Memory sequence predictor applied in the spatiotemporal forecasting setting. Each spatial node is modeled independently as a univariate sequence, with the LSTM gates learning to selectively retain or forget information across timesteps — providing a simple but effective recurrent baseline for node-structured time series data.

## Paper
- **Title**: Long Short-Term Memory
- **Venue**: Neural Computation 1997
- **Published**: 1997
- **arXiv**: N/A

## Abstract
Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. Local in space and time; their computational complexity per time step and weight is O(1). Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.

## In ModernTSF
Default config: `configs/models/LSTM.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{hochreiter1997long,
  author  = {Sepp Hochreiter and J{\"u}rgen Schmidhuber},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  doi     = {10.1162/neco.1997.9.8.1735},
  url     = {https://doi.org/10.1162/neco.1997.9.8.1735}
}
```
