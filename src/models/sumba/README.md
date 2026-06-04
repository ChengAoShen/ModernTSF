---
model: "Sumba"
forecasting_setting: "time_series"
config: "configs/models/Sumba.toml"
registry: "models.sumba.registry"
paper_title: "Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics"
venue: "NeurIPS 2024"
year: 2024
arxiv: ""
---
# Sumba

Sumba is a time series forecasting model for multivariate sequences that directly parameterizes spatial structures using a learnable matrix basis and a convex combination. Its dynamic spatial structure generation function operates within a well-constrained output space, producing lower-variance graph structures with interpretable dynamics, and combines dilated inception temporal convolution blocks with dynamic graph convolution to jointly model temporal dependencies and inter-variate correlations.

## Paper
- **Title**: Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics
- **Venue**: NeurIPS 2024
- **Published**: 2024
- **arXiv**: N/A

## Abstract
Multivariate time series forecasting is of central importance in modern intelligent decision systems. The dynamics of multivariate time series are jointly characterized by temporal dependencies and spatial correlations. Hence, it is equally important to build the forecasting models from both perspectives. The real-world multivariate time series data often presents spatial correlations that show structures and evolve dynamically. To capture such dynamic spatial structures, the existing forecasting approaches often rely on a two-stage learning process (learning dynamic series representations and then generating spatial structures), which is sensitive to the small time-window input data and has high variance. To address this, we propose a novel forecasting model with a structured matrix basis. At its core is a dynamic spatial structure generation function whose output space is well-constrained and the generated structures have lower variance, meanwhile, it is more expressive and can offer interpretable dynamics. This is achieved through a novel structured parameterization and imposing structure regularization on the matrix basis. Extensive experiments on six benchmark datasets demonstrate that our model achieves up to 8.5% improvements over the existing methods, while providing interpretability into the underlying system dynamics.

## In ModernTSF
Default config: `configs/models/Sumba.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
