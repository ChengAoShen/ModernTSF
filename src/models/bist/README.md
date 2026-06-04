---
model: "BiST"
forecasting_setting: "spatiotemporal"
config: "configs/models/BiST.toml"
registry: "models.bist.registry"
paper_title: "BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction"
venue: "PVLDB 2025"
year: 2025
arxiv: ""
---
# BiST

BiST is a spatiotemporal learning model for node-structured or graph-structured data that simultaneously captures temporal dynamics and spatial relationships between nodes. It challenges the standard input-label spatiotemporal consistency assumption by incorporating label information during training via a lightweight bidirectional MLP backbone with an adaptive graph, enabling strong predictive performance with a fraction of the training time and memory of existing methods.

## Paper
- **Title**: BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction
- **Venue**: Proceedings of the VLDB Endowment (PVLDB), Vol. 18, No. 6
- **Published**: 2025
- **arXiv**: N/A

## Abstract
While existing spatiotemporal prediction models have shown promising performance, they often rely on the assumption of input-label spatiotemporal consistency, and their high complexity raises concerns about scalability. BiST addresses these issues by decomposing the prediction into a forward spatiotemporal learning process that generates base predictions and a residual correction process that models spatiotemporal residuals to refine those predictions. The backbone is a lightweight MLP rather than stacked spatiotemporal layers, yielding competitive accuracy while consuming only a small fraction of the training time and memory of state-of-the-art models.

## In ModernTSF
Default config: `configs/models/BiST.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
