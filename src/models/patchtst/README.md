---
model: "PatchTST"
forecasting_setting: "time_series"
config: "configs/models/PatchTST.toml"
registry: "models.patchtst.registry"
paper_title: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
venue: "ICLR 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2211.14730"
---
# PatchTST

PatchTST is a Transformer-based model for multivariate and univariate long-term time-series forecasting that segments each channel into subseries-level patches fed as input tokens, combined with a channel-independence strategy where each channel shares the same Transformer weights. This design retains local semantic information, drastically reduces attention-map memory, and allows the model to attend over a much longer historical context.

## Paper
- **Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Venue**: ICLR 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.14730

## Abstract
We propose an efficient design of Transformer-based models for multivariate time series forecasting and self-supervised representation learning. It is based on two key components: (i) segmentation of time series into subseries-level patches which are served as input tokens to Transformer; (ii) channel-independence where each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series. Patching design naturally has three-fold benefit: local semantic information is retained in the embedding; computation and memory usage of the attention maps are quadratically reduced given the same look-back window; and the model can attend longer history. Our channel-independent patch time series Transformer (PatchTST) can improve the long-term forecasting accuracy significantly when compared with that of SOTA Transformer-based models. We also apply our model to self-supervised pre-training tasks and attain excellent fine-tuning performance, which outperforms supervised training on large datasets. Transferring of masked pre-trained representation on one dataset to others also produces SOTA forecasting accuracy.

## In ModernTSF
Default config: `configs/models/PatchTST.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/NieNSK23,
  author       = {Yuqi Nie and
                  Nam H. Nguyen and
                  Phanwadee Sinthong and
                  Jayant Kalagnanam},
  title        = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/forum?id=Jbdc0vTOcol},
  timestamp    = {Wed, 24 Jul 2024 16:50:33 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/NieNSK23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
