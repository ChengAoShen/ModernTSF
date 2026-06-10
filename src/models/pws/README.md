---
model: "PWS"
forecasting_setting: "time_series"
config: "configs/models/PWS.toml"
registry: "models.pws.registry"
paper_title: ""
venue: "N/A (simple in-repo baseline)"
arxiv: ""
---
# PWS

PWS (Patch Weighted Sum) is a deliberately minimal in-repo baseline for univariate and multivariate time-series forecasting. It splits the look-back window period-wise into fixed-size patches and produces the forecast as a learned weighted sum over those patches — a per-patch linear map from historical periods to future periods, with an optional RevIN normalisation layer. There is no attention, convolution, or non-linear backbone; it exists purely as a lightweight reference point.

## Paper
PWS (Patch Weighted Sum) has no associated publication. It is a deliberately simple baseline implemented directly in ModernTSF — no vendored upstream and no external paper.
- **Venue**: N/A (simple in-repo baseline)
- **arXiv**: N/A

## Abstract
PWS is the simplest possible patch baseline. It partitions the look-back window into complete periods of a user-specified length, divides each period into non-overlapping patches, and for each patch position learns a linear weighted sum that maps historical periods to future periods. The final prediction is assembled by concatenating the per-patch outputs across the period and trimming to the prediction horizon. An optional RevIN normalisation layer handles distributional shift. With no attention or convolution operators, it serves as a lightweight reference baseline rather than a novel architecture.

## In ModernTSF
Default config: `configs/models/PWS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

PWS is an in-repository baseline and has no associated paper or canonical
BibTeX entry.
