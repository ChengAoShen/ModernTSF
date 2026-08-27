---
name: "AirFormer"
implementation: rewrite
summary: "The AirFormer paper combines causal temporal attention, dartboard spatial attention, and top-down stochastic latent variables for nationwide air-quality forecasting. This ModernTSF entry consumes historical values and time marks, retains the temporal and stochastic paths, and returns point forecasts, but disables the dataset-specific dartboard spatial path and does not consume known future covariates."
paper:
  title: "AirFormer: Predicting Nationwide Air Quality in China with Transformers"
  venue: "AAAI 2023"
  year: 2023
  url: "https://doi.org/10.1609/aaai.v37i12.26676"
codebase:
  url: "https://github.com/yoshall/airformer"
  revision: "ef7d3933768490e3a06921b8eb0f837c61741194"
  license: ""
  usage: reference-only
---
# AirFormer

The AirFormer paper combines causal temporal attention, dartboard spatial attention, and top-down stochastic latent variables for nationwide air-quality forecasting. This ModernTSF entry consumes historical values and time marks, retains the temporal and stochastic paths, and returns point forecasts, but disables the dataset-specific dartboard spatial path and does not consume known future covariates.

## Paper
- **Title**: AirFormer: Predicting Nationwide Air Quality in China with Transformers
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.15979

## Abstract
Air pollution is a crucial issue affecting human health and livelihoods, as well as one of the barriers to economic and social growth. Forecasting air quality has become an increasingly important endeavor with significant social impacts, especially in emerging countries like China. In this paper, we present a novel Transformer architecture termed AirFormer to collectively predict nationwide air quality in China, with an unprecedented fine spatial granularity covering thousands of locations. AirFormer decouples the learning process into two stages -- 1) a bottom-up deterministic stage that contains two new types of self-attention mechanisms to efficiently learn spatio-temporal representations; 2) a top-down stochastic stage with latent variables to capture the intrinsic uncertainty of air quality data. We evaluate AirFormer with 4-year data from 1,085 stations in the Chinese Mainland. Compared to the state-of-the-art model, AirFormer reduces prediction errors by 5%~8% on 72-hour future predictions. Our source code is available at this https URL.

## In ModernTSF
Default config: `configs/models/AirFormer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Source and verification

- Official source: https://github.com/yoshall/airformer at `ef7d3933768490e3a06921b8eb0f837c61741194` (no license file declared at that revision).
Implementation: **rewrite** (clean-room audit pending). The implementation was consolidated from CauAir's baseline and no numerical parity result is recorded.
- Known differences: dataset-specific dartboard partitions and DS-MSA are disabled in generic mode, which replaces them with 1x1 residual projections. The adapter returns point forecasts only and omits the official reconstruction output and KL-divergence training term; generic time marks replace the original data pipeline.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/LiangXKWWZZZ23,
  author       = {Yuxuan Liang and
                  Yutong Xia and
                  Songyu Ke and
                  Yiwei Wang and
                  Qingsong Wen and
                  Junbo Zhang and
                  Yu Zheng and
                  Roger Zimmermann},
  editor       = {Brian Williams and
                  Yiling Chen and
                  Jennifer Neville},
  title        = {AirFormer: Predicting Nationwide Air Quality in China with Transformers},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2023, Thirty-Fifth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2023, Thirteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2023, Washington, DC, USA, February
                  7-14, 2023},
  pages        = {14329--14337},
  publisher    = {{AAAI} Press},
  year         = {2023},
  url          = {https://doi.org/10.1609/aaai.v37i12.26676},
  doi          = {10.1609/AAAI.V37I12.26676},
  timestamp    = {Wed, 18 Mar 2026 17:07:12 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/LiangXKWWZZZ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
