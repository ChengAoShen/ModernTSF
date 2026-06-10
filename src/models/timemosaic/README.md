---
model: "TimeMosaic"
forecasting_setting: "time_series"
config: "configs/models/TimeMosaic.toml"
registry: "models.timemosaic.registry"
paper_title: "TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding"
venue: "AAAI 2026"
year: 2026
arxiv: "https://arxiv.org/abs/2509.19406"
---
# TimeMosaic

TimeMosaic is a time-series forecasting model designed to handle temporal heterogeneity in multivariate data. It employs adaptive patch embedding to dynamically adjust segmentation granularity based on local information density, and a segment-wise decoder that treats each prediction horizon as a related but distinct sub-task, adapting to horizon-specific difficulty rather than applying a single uniform decoder.

## Paper
- **Title**: TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding
- **Venue**: AAAI 2026
- **Published**: 2026 (arXiv: 2025-09)
- **arXiv**: https://arxiv.org/abs/2509.19406

## Abstract
Multivariate time series forecasting is essential in domains such as finance, transportation, climate, and energy. However, existing patch-based methods typically adopt fixed-length segmentation, overlooking the heterogeneity of local temporal dynamics and the decoding heterogeneity of forecasting. Such designs lose details in information-dense regions, introduce redundancy in stable segments, and fail to capture the distinct complexities of short-term and long-term horizons. We propose TimeMosaic, a forecasting framework that aims to address temporal heterogeneity. TimeMosaic employs adaptive patch embedding to dynamically adjust granularity according to local information density, balancing motif reuse with structural clarity while preserving temporal continuity. In addition, it introduces segment-wise decoding that treats each prediction horizon as a related subtask and adapts to horizon-specific difficulty and information requirements, rather than applying a single uniform decoder. Extensive evaluations on benchmark datasets demonstrate that TimeMosaic delivers consistent improvements over existing methods, and our model trained on the large-scale corpus with 321 billion observations achieves performance competitive with state-of-the-art TSFMs.

## In ModernTSF
Default config: `configs/models/TimeMosaic.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/DingFHWWYZ26,
  author       = {Kuiye Ding and
                  Fanda Fan and
                  Chunyi Hou and
                  Zheya Wang and
                  Lei Wang and
                  Zhengxin Yang and
                  Jianfeng Zhan},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting
                  via Adaptive Granularity Patch and Segment-wise Decoding},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {20790--20798},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i25.39218},
  doi          = {10.1609/AAAI.V40I25.39218},
  timestamp    = {Fri, 27 Mar 2026 17:13:39 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/DingFHWWYZ26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
