---
model: "KNNForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/KNNForecasterTS.toml"
registry: "models.knn_forecaster_ts.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# KNNForecasterTS

KNNForecasterTS is a differentiable k-nearest-neighbours style forecaster for the standard univariate and multivariate time-series setting. Instead of a hard discrete lookup, it uses a set of learnable prototype vectors and RBF (radial basis function) kernel weights to produce a soft weighted combination of prototypes, making the entire prediction end-to-end trainable with gradient descent and compatible with GPU acceleration via PyTorch.

## Paper
- **Title**: N/A (classical baseline)
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
K-nearest neighbours (KNN) regression is a non-parametric method that predicts an output by averaging the target values of the k training samples closest (in feature space) to the query point, using a distance metric such as Euclidean distance. Applied to time-series forecasting, KNN finds the k historical windows most similar to the current input window and uses their corresponding future segments as the forecast. The method has no single defining paper; it originates from the general KNN algorithm described by Fix & Hodges (1951) and Cover & Hart (1967). In ModernTSF, KNNForecasterTS replaces the hard discrete lookup with differentiable RBF-weighted prototypes so the model can be trained end-to-end with the standard gradient-based trainer and can run on CUDA/MPS devices.

## In ModernTSF
Default config: `configs/models/KNNForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{cover1967nearest,
  author  = {Thomas M. Cover and Peter E. Hart},
  title   = {Nearest Neighbor Pattern Classification},
  journal = {IEEE Transactions on Information Theory},
  volume  = {13},
  number  = {1},
  pages   = {21--27},
  year    = {1967},
  doi     = {10.1109/TIT.1967.1053964}
}
```
