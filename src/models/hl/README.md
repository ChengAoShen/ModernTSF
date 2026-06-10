---
model: "HL"
forecasting_setting: "spatiotemporal"
config: "configs/models/HL.toml"
registry: "models.hl.registry"
paper_title: ""
venue: "N/A (classical baseline)"
arxiv: ""
---
# HL

HL (Historical Last) is a naive spatiotemporal forecasting baseline that repeats the last observed value across every node and every step of the prediction horizon. It serves as a lower-bound reference in graph- and node-structured benchmarks, providing the simplest possible prediction without any learning.

## Paper
- **Title**: N/A
- **Venue**: N/A (classical baseline)
- **Published**: N/A
- **arXiv**: N/A

## Abstract
Historical Last (HL) is a naive persistence baseline for spatiotemporal forecasting. For each node in the graph, it copies the final observed value from the input window and repeats it identically across all future time steps in the prediction horizon. Despite its simplicity, the method provides a meaningful lower bound: any learned model should outperform HL, especially over longer horizons where temporal dynamics diverge from the last observation. It requires no training and has no learnable parameters.

## In ModernTSF
Default config: `configs/models/HL.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

HL is an in-repository persistence baseline and has no associated paper or
canonical BibTeX entry.
