---
model: "PatchMLP"
forecasting_setting: "time_series"
config: "configs/models/PatchMLP.toml"
registry: "models.patchmlp.registry"
paper_title: "Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2405.13575"
---
# PatchMLP

PatchMLP is a patch-based MLP model for long-term time series forecasting that attributes the effectiveness of recent Transformer models to their patch mechanism rather than to attention. It applies moving-average decomposition to separate smooth trend components from noise residuals, then processes the smooth branch with cross-variable channel mixing for semantic information exchange and handles the residual branch with channel-independent linear layers, achieving competitive accuracy without any attention operations.

## Paper
- **Title**: Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2024-05)
- **arXiv**: https://arxiv.org/abs/2405.13575

## Abstract
Recent studies have attempted to refine the Transformer architecture to demonstrate its effectiveness in Long-Term Time Series Forecasting (LTSF) tasks. Despite surpassing many linear forecasting models with ever-improving performance, we remain skeptical of Transformers as a solution for LTSF. We attribute the effectiveness of these models largely to the adopted Patch mechanism, which enhances sequence locality to an extent yet fails to fully address the loss of temporal information inherent to the permutation-invariant self-attention mechanism. Further investigation suggests that simple linear layers augmented with the Patch mechanism may outperform complex Transformer-based LTSF models. Moreover, diverging from models that use channel independence, our research underscores the importance of cross-variable interactions in enhancing the performance of multivariate time series forecasting. The interaction information between variables is highly valuable but has been misapplied in past studies, leading to suboptimal cross-variable models. Based on these insights, we propose a novel and simple Patch-based MLP (PatchMLP) for LTSF tasks. Specifically, we employ simple moving averages to extract smooth components and noise-containing residuals from time series data, engaging in semantic information interchange through channel mixing and specializing in random noise with channel independence processing. The PatchMLP model consistently achieves state-of-the-art results on several real-world datasets. We hope this surprising finding will spur new research directions in the LTSF field and pave the way for more efficient and concise solutions.

## In ModernTSF
Default config: `configs/models/PatchMLP.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/TangZ25,
  author       = {Peiwang Tang and
                  Weitai Zhang},
  editor       = {Toby Walsh and
                  Julie Shah and
                  Zico Kolter},
  title        = {Unlocking the Power of Patch: Patch-Based {MLP} for Long-Term Time
                  Series Forecasting},
  booktitle    = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, Thirty-Seventh
                  Conference on Innovative Applications of Artificial Intelligence,
                  Fifteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2025, Philadelphia, PA, USA, February 25 - March 4, 2025},
  pages        = {12640--12648},
  publisher    = {{AAAI} Press},
  year         = {2025},
  url          = {https://doi.org/10.1609/aaai.v39i12.33378},
  doi          = {10.1609/AAAI.V39I12.33378},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/TangZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
