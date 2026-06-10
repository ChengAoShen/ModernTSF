---
model: "TCNForecasterTS"
forecasting_setting: "time_series"
config: "configs/models/TCNForecasterTS.toml"
registry: "models.tcn_forecaster_ts.registry"
paper_title: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
venue: "arXiv preprint"
year: 2018
arxiv: "https://arxiv.org/abs/1803.01271"
---
# TCNForecasterTS

TCNForecasterTS is a compact Temporal Convolutional Network (TCN) forecaster registered as a neural baseline in ModernTSF for the standard time-series forecasting setting. It implements the dilated causal convolutional architecture with residual connections from Bai et al. (2018), adapted as a PyTorch-native adapter using the standard ModernTSF trainer interface.

## Paper
- **Title**: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **Venue**: arXiv preprint
- **Published**: 2018
- **arXiv**: https://arxiv.org/abs/1803.01271

## Abstract
For most deep learning practitioners, sequence modeling is synonymous with recurrent networks. Yet recent results indicate that convolutional architectures can outperform recurrent networks on tasks such as audio synthesis and machine translation. Given a new sequence modeling task or dataset, which architecture should one use? We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling. The models are evaluated across a broad range of standard tasks that are commonly used to benchmark recurrent networks. Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks. To assist related work, we have made code available at http://github.com/locuslab/TCN.

## In ModernTSF
Default config: `configs/models/TCNForecasterTS.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-1803-01271,
  author       = {Shaojie Bai and
                  J. Zico Kolter and
                  Vladlen Koltun},
  title        = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks
                  for Sequence Modeling},
  journal      = {CoRR},
  volume       = {abs/1803.01271},
  year         = {2018},
  url          = {http://arxiv.org/abs/1803.01271},
  eprinttype   = {arXiv},
  eprint       = {1803.01271},
  timestamp    = {Mon, 13 Aug 2018 16:47:39 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1803-01271.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
