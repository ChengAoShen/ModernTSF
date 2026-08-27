---
name: "TCNForecasterTS"
implementation: rewrite
summary: "TCNForecasterTS is a compact Temporal Convolutional Network (TCN) forecaster registered as a neural baseline in ModernTSF for the standard time-series forecasting setting. It implements the dilated causal convolutional architecture with residual connections from Bai et al. (2018), adapted as a PyTorch-native adapter using the standard ModernTSF trainer interface."
paper:
  title: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
  venue: "arXiv preprint"
  year: 2018
  url: "https://arxiv.org/abs/1803.01271"
codebase:
  url: ""
  revision: ""
  license: ""
  usage: none
---
# TCNForecasterTS

TCNForecasterTS is a compact Temporal Convolutional Network (TCN) forecaster registered as a neural baseline in ModernTSF for the standard time-series forecasting setting. It implements the dilated causal convolutional architecture with residual connections from Bai et al. (2018), adapted as a PyTorch-native adapter using the standard ModernTSF trainer interface.

<!-- model-card:canonical:start -->
## Method overview

TCNForecasterTS is a compact Temporal Convolutional Network (TCN) forecaster registered as a neural baseline in ModernTSF for the standard time-series forecasting setting.

## Core architecture

It implements the dilated causal convolutional architecture with residual connections from Bai et al. (2018), adapted as a PyTorch-native adapter using the standard ModernTSF trainer interface.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://arxiv.org/abs/1803.01271); title: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling; venue/year: arXiv preprint / 2018
- codebase: not available; revision: `not available`; license: `not available`; usage: `none`

## Local implementation

This card declares a `rewrite` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/TCNForecasterTS.toml`](../../../configs/models/TCNForecasterTS.toml).

## Differences

No additional implementation differences are recorded in the preserved card notes. This is an explicit documentation gap, not an equivalence claim.

## Shared components

No cataloged shared component is imported; the architecture remains model-local.

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=64`, `dropout=0.1`, `num_layers=1`, `num_estimators=16`, `tree_depth=3`, `num_prototypes=32`, `kernel_gamma=0.1`, `l1_penalty=0.0`, `l2_penalty=0.0`, `use_revin=True`
<!-- model-card:canonical:end -->

## Paper
- **Title**: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
- **Venue**: arXiv preprint
- **Published**: 2018
- **arXiv**: https://arxiv.org/abs/1803.01271

## Abstract
For most deep learning practitioners, sequence modeling is synonymous with recurrent networks. Yet recent results indicate that convolutional architectures can outperform recurrent networks on tasks such as audio synthesis and machine translation. Given a new sequence modeling task or dataset, which architecture should one use? We conduct a systematic evaluation of generic convolutional and recurrent architectures for sequence modeling. The models are evaluated across a broad range of standard tasks that are commonly used to benchmark recurrent networks. Our results indicate that a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory. We conclude that the common association between sequence modeling and recurrent networks should be reconsidered, and convolutional networks should be regarded as a natural starting point for sequence modeling tasks. To assist related work, we have made code available at http://github.com/locuslab/TCN.

## In ModernTSF
Default config: `configs/models/TCNForecasterTS.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

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
