---
model: "RPMixer"
forecasting_setting: "spatiotemporal"
config: "configs/models/RPMixer.toml"
spec: "models.rpmixer.spec"
paper_title: "RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data"
venue: "KDD 2024"
year: 2024
arxiv: "https://arxiv.org/abs/2402.10487"
---
# RPMixer

RPMixer is a spatiotemporal forecasting model built on an all-MLP (all-Multi-Layer Perceptron) architecture that forgoes explicit graph-based spatial modeling in favour of general time series mixing. It addresses the tendency of standard MLP-mixer models to overfit on large-scale spatial-temporal datasets by inserting random projection layers between blocks to increase output diversity, exploiting the ensemble-like behaviour of deep residual networks where each block acts as a base learner. The approach achieves competitive or superior performance against both graph-based and general forecasting baselines on large spatial-temporal benchmarks.

## Paper
- **Title**: RPMixer: Shaking Up Time Series Forecasting with Random Projections for Large Spatial-Temporal Data
- **Venue**: KDD 2024
- **Published**: 2024 (arXiv: 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.10487

## Abstract
Spatial-temporal forecasting systems play a crucial role in addressing numerous real-world challenges. In this paper, we investigate the potential of addressing spatial-temporal forecasting problems using general time series forecasting models, i.e., models that do not leverage the spatial relationships among the nodes. We propose a all-Multi-Layer Perceptron (all-MLP) time series forecasting architecture called RPMixer. The all-MLP architecture was chosen due to its recent success in time series forecasting benchmarks. Furthermore, our method capitalizes on the ensemble-like behavior of deep neural networks, where each individual block within the network behaves like a base learner in an ensemble model, particularly when identity mapping residual connections are incorporated. By integrating random projection layers into our model, we increase the diversity among the blocks' outputs, thereby improving the overall performance of the network. Extensive experiments conducted on the largest spatial-temporal forecasting benchmark datasets demonstrate that the proposed method outperforms alternative methods, including both spatial-temporal graph models and general forecasting models.

## In ModernTSF
Default config: `configs/models/RPMixer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Evidence level: **unverified**. The paper provides no official code release.
The immediate implementation source is
[`PoorOtterBob/CauAir`](https://github.com/PoorOtterBob/CauAir) revision
`73dae00ca6ad14abb15174a0a0286d500e868b94`, whose repository declares no
license. The all-MLP residual stack, frozen random projections, frequency-domain
mixing, and reversible normalization are present, but numerical equivalence,
benchmark feature construction, and the published optimization protocol cannot
be established. Previously exposed `IE_dim`, `dropout`, and `num_head` settings
were removed because they never affected the implementation.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/YehFDSLAWC0ZWZ24,
  author       = {Chin{-}Chia Michael Yeh and
                  Yujie Fan and
                  Xin Dai and
                  Uday Singh Saini and
                  Vivian Lai and
                  Prince Osei Aboagye and
                  Junpeng Wang and
                  Huiyuan Chen and
                  Yan Zheng and
                  Zhongfang Zhuang and
                  Liang Wang and
                  Wei Zhang},
  editor       = {Ricardo Baeza{-}Yates and
                  Francesco Bonchi},
  title        = {RPMixer: Shaking Up Time Series Forecasting with Random Projections
                  for Large Spatial-Temporal Data},
  booktitle    = {Proceedings of the 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2024, Barcelona, Spain, August 25-29, 2024},
  pages        = {3919--3930},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3637528.3671881},
  doi          = {10.1145/3637528.3671881},
  timestamp    = {Mon, 24 Nov 2025 11:08:37 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/YehFDSLAWC0ZWZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
