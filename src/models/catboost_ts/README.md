---
model: "CatBoostTS"
forecasting_setting: "time_series"
config: "configs/models/CatBoostTS.toml"
registry: "models.catboost_ts.registry"
paper_title: "CatBoost: unbiased boosting with categorical features"
venue: "NeurIPS 2018"
year: 2018
arxiv: "https://arxiv.org/abs/1706.09516"
---
# CatBoostTS

CatBoostTS is a time-series forecasting adapter built around the CatBoost gradient-boosting algorithm, applied to the standard time-series forecasting setting. It accepts a fixed-length historical window of multivariate numerical values and produces a fixed-length forecast horizon, using the ordered boosting and categorical-feature-processing techniques introduced in the CatBoost paper.

## Paper
- **Title**: CatBoost: unbiased boosting with categorical features
- **Venue**: NeurIPS 2018
- **Published**: 2018 (arXiv: 2017-06)
- **arXiv**: https://arxiv.org/abs/1706.09516

## Abstract
This paper presents the key algorithmic techniques behind CatBoost, a new gradient boosting toolkit. Their combination leads to CatBoost outperforming other publicly available boosting implementations in terms of quality on a variety of datasets. Two critical algorithmic advances introduced in CatBoost are the implementation of ordered boosting, a permutation-driven alternative to the classic algorithm, and an innovative algorithm for processing categorical features. Both techniques were created to fight a prediction shift caused by a special kind of target leakage present in all currently existing implementations of gradient boosting algorithms. In this paper, we provide a detailed analysis of this problem and demonstrate that proposed algorithms solve it effectively, leading to excellent empirical results.

## In ModernTSF
Default config: `configs/models/CatBoostTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/ProkhorenkovaGV18,
  author       = {Liudmila Ostroumova Prokhorenkova and
                  Gleb Gusev and
                  Aleksandr Vorobev and
                  Anna Veronika Dorogush and
                  Andrey Gulin},
  editor       = {Samy Bengio and
                  Hanna M. Wallach and
                  Hugo Larochelle and
                  Kristen Grauman and
                  Nicol{\`{o}} Cesa{-}Bianchi and
                  Roman Garnett},
  title        = {CatBoost: unbiased boosting with categorical features},
  booktitle    = {Advances in Neural Information Processing Systems 31: Annual Conference
                  on Neural Information Processing Systems 2018, NeurIPS 2018, December
                  3-8, 2018, Montr{\'{e}}al, Canada},
  pages        = {6639--6649},
  year         = {2018},
  url          = {https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html},
  timestamp    = {Mon, 16 May 2022 15:41:51 +0200},
  biburl       = {https://dblp.org/rec/conf/nips/ProkhorenkovaGV18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
