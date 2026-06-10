---
model: "LightGBMTS"
forecasting_setting: "time_series"
config: "configs/models/LightGBMTS.toml"
registry: "models.lightgbm_ts.registry"
paper_title: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
venue: "NeurIPS 2017"
year: 2017
arxiv: ""
---
# LightGBMTS

LightGBMTS is a PyTorch-native adapter that brings the LightGBM-style lightweight gradient boosting approach to multivariate time series forecasting. It implements a residual ensemble of soft decision trees with gradient-boosted updates applied to lagged-window inputs, running through the standard ModernTSF trainer on CPU, CUDA, or MPS devices.

## Paper
- **Title**: LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- **Venue**: NeurIPS 2017
- **Published**: 2017
- **arXiv**: N/A

## Abstract
Gradient Boosting Decision Tree (GBDT) is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large. A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming. To tackle this problem, we propose two novel techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain. We prove that, since the data instances with larger gradients play a more important role in the computation of information gain, GOSS can obtain quite accurate estimation of the information gain with a much smaller data size. With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features. We prove that finding the optimal bundling of exclusive features is NP-hard, but a greedy algorithm can achieve quite good approximation ratio (and thus can effectively reduce the number of features without hurting the accuracy of split point determination by much). We call our new GBDT implementation with GOSS and EFB LightGBM. Our experiments on multiple public datasets show that, LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.

## In ModernTSF
Default config: `configs/models/LightGBMTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/KeMFWCMYL17,
  author       = {Guolin Ke and
                  Qi Meng and
                  Thomas Finley and
                  Taifeng Wang and
                  Wei Chen and
                  Weidong Ma and
                  Qiwei Ye and
                  Tie{-}Yan Liu},
  editor       = {Isabelle Guyon and
                  Ulrike von Luxburg and
                  Samy Bengio and
                  Hanna M. Wallach and
                  Rob Fergus and
                  S. V. N. Vishwanathan and
                  Roman Garnett},
  title        = {LightGBM: {A} Highly Efficient Gradient Boosting Decision Tree},
  booktitle    = {Advances in Neural Information Processing Systems 30: Annual Conference
                  on Neural Information Processing Systems 2017, December 4-9, 2017,
                  Long Beach, CA, {USA}},
  pages        = {3146--3154},
  year         = {2017},
  url          = {https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html},
  timestamp    = {Thu, 21 Jan 2021 15:15:21 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/KeMFWCMYL17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
