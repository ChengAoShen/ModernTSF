---
name: "PWS"
implementation: rewrite
summary: "PWS (Patch Weighted Sum) is a deliberately minimal in-repo baseline for univariate and multivariate time-series forecasting. It splits the look-back window period-wise into fixed-size patches, refines each patch with a small analysis MLP, and produces the forecast with a learned map from historical periods to future periods. It has optional RevIN normalization but no attention or convolution."
paper:
  title: "Patch Weighted Sum (ModernTSF baseline)"
  venue: "ModernTSF"
  year: 2026
  url: ""
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# PWS

PWS (Patch Weighted Sum) is a deliberately minimal in-repo baseline for univariate and multivariate time-series forecasting. It splits the look-back window period-wise into fixed-size patches, refines each patch with a small analysis MLP, and produces the forecast with a learned map from historical periods to future periods. It has optional RevIN normalization but no attention or convolution.

## Paper
PWS (Patch Weighted Sum) has no associated publication. It is a deliberately simple baseline implemented directly in ModernTSF — no vendored upstream and no external paper.
- **Venue**: N/A (simple in-repo baseline)
- **arXiv**: N/A

## Abstract
PWS partitions the look-back window into complete periods of a user-specified length and divides each period into non-overlapping patches. A patch-specific residual MLP analyzes the historical-period axis, followed by a learned linear map to future periods. The final prediction concatenates patch outputs across the period and trims to the horizon. Optional RevIN handles distribution shift.

## In ModernTSF
Default config: `configs/models/PWS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Implementation: `rewrite` (clean-room audit pending). PWS is an intentional ModernTSF baseline with no external paper or upstream repository.
- `analysis_hidden` is a typed list and `analysis_act` accepts only implemented activations, preventing silent no-op configurations.

## Citation

PWS is an in-repository baseline and has no associated paper or canonical
BibTeX entry.
