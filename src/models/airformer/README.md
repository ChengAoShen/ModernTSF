---
model: "AirFormer"
forecasting_setting: "covariate"
config: "configs/models/AirFormer.toml"
registry: "models.airformer.registry"
paper_title: "AirFormer: Predicting Nationwide Air Quality in China with Transformers"
venue: "AAAI 2023"
year: 2023
arxiv: "https://arxiv.org/abs/2211.15979"
---
# AirFormer

AirFormer is a covariate prediction model designed for nationwide air quality forecasting. It targets node-level value prediction and leverages both historical covariates and known future covariates. The architecture decouples learning into a bottom-up deterministic stage that uses two novel self-attention mechanisms to capture spatio-temporal representations, and a top-down stochastic stage with latent variables that models the intrinsic uncertainty of air quality data.

## Paper
- **Title**: AirFormer: Predicting Nationwide Air Quality in China with Transformers
- **Venue**: AAAI 2023
- **Published**: 2023 (arXiv: 2022-11)
- **arXiv**: https://arxiv.org/abs/2211.15979

## Abstract
Air pollution is a crucial issue affecting human health and livelihoods, as well as one of the barriers to economic and social growth. Forecasting air quality has become an increasingly important endeavor with significant social impacts, especially in emerging countries like China. In this paper, we present a novel Transformer architecture termed AirFormer to collectively predict nationwide air quality in China, with an unprecedented fine spatial granularity covering thousands of locations. AirFormer decouples the learning process into two stages -- 1) a bottom-up deterministic stage that contains two new types of self-attention mechanisms to efficiently learn spatio-temporal representations; 2) a top-down stochastic stage with latent variables to capture the intrinsic uncertainty of air quality data. We evaluate AirFormer with 4-year data from 1,085 stations in the Chinese Mainland. Compared to the state-of-the-art model, AirFormer reduces prediction errors by 5%~8% on 72-hour future predictions. Our source code is available at this https URL.

## In ModernTSF
Default config: `configs/models/AirFormer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

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
