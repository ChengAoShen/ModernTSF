---
model: "SVRForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/SVRForecasterTS.toml"
registry: "models.svr_forecaster_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# SVRForecasterTS

SVRForecasterTS is a PyTorch-native time series forecasting adapter inspired by Support Vector Regression (SVR). It uses RBF (radial basis function) prototype support vectors and a linear residual head to produce multi-step forecasts, wrapped in the standard ModernTSF `torch.nn.Module` interface so it can be trained with gradient descent and run on CPU, CUDA, or MPS hardware alongside deep learning models.

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Support Vector Regression (SVR) is a classical kernel-based supervised learning method derived from Support Vector Machines. Given a set of training examples, SVR seeks a function that deviates from the true target values by at most a margin epsilon while remaining as flat as possible. Predictions are expressed as a weighted sum of kernel evaluations (commonly the RBF kernel) between the query point and a sparse subset of training examples called support vectors. SVRForecasterTS re-implements this kernel regression idea as a differentiable PyTorch module: learnable RBF prototype centers replace the classical SVM solver, and a linear residual layer corrects systematic bias. This allows the classical SVR concept to be trained end-to-end with gradient descent and evaluated on GPU or MPS hardware within the ModernTSF benchmark pipeline.

## In ModernTSF
Default config: `configs/models/SVRForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{drucker1996support,
  author    = {Harris Drucker and Christopher J. C. Burges and Linda Kaufman and Alexander J. Smola and Vladimir Vapnik},
  title     = {Support Vector Regression Machines},
  booktitle = {Advances in Neural Information Processing Systems 9 (NIPS 1996)},
  pages     = {155--161},
  year      = {1996},
  url       = {https://proceedings.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html}
}
```
