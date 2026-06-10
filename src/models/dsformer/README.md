---
model: "DSFormer"
forecasting_setting: "time_series"
config: "configs/models/DSFormer.toml"
registry: "models.dsformer.registry"
paper_title: "DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction"
venue: "CIKM 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2308.03274"
---
# DSFormer

DSFormer (Double Sampling Transformer) is a Transformer-based model for multivariate long-term time series forecasting. It combines a Double Sampling (DS) block — which applies down-sampling and piecewise sampling to capture global and local temporal information — with a Temporal Variable Attention (TVA) block that mines both temporal and inter-variable dependencies, feeding a generative MLP decoder to produce multi-horizon forecasts.

## Paper
- **Title**: DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction
- **Venue**: CIKM 2023
- **Published**: 2023 (arXiv: 2023-08)
- **arXiv**: https://arxiv.org/abs/2308.03274

## Abstract
Multivariate time series long-term prediction, which aims to predict the change of data in a long time, can provide references for decision-making. Although transformer-based models have made progress in this field, they usually do not make full use of three features of multivariate time series: global information, local information, and variables correlation. To effectively mine the above three features and establish a high-precision prediction model, we propose a double sampling transformer (DSformer), which consists of the double sampling (DS) block and the temporal variable attention (TVA) block. Firstly, the DS block employs down sampling and piecewise sampling to transform the original series into feature vectors that focus on global information and local information respectively. Then, TVA block uses temporal attention and variable attention to mine these feature vectors from different dimensions and extract key information. Finally, based on a parallel structure, DSformer uses multiple TVA blocks to mine and integrate different features obtained from DS blocks respectively. The integrated feature information is passed to the generative decoder based on a multi-layer perceptron to realize multivariate time series long-term prediction. Experimental results on nine real-world datasets show that DSformer can outperform eight existing baselines.

## In ModernTSF
Default config: `configs/models/DSFormer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/cikm/YuWSSWX23,
  author       = {Chengqing Yu and
                  Fei Wang and
                  Zezhi Shao and
                  Tao Sun and
                  Lin Wu and
                  Yongjun Xu},
  editor       = {Ingo Frommholz and
                  Frank Hopfgartner and
                  Mark Lee and
                  Michael Oakes and
                  Mounia Lalmas and
                  Min Zhang and
                  Rodrygo L. T. Santos},
  title        = {DSformer: {A} Double Sampling Transformer for Multivariate Time Series
                  Long-term Prediction},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom,
                  October 21-25, 2023},
  pages        = {3062--3072},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3583780.3614851},
  doi          = {10.1145/3583780.3614851},
  timestamp    = {Mon, 10 Feb 2025 16:22:03 +0100},
  biburl       = {https://dblp.org/rec/conf/cikm/YuWSSWX23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
