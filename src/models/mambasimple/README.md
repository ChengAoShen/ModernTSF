---
model: "MambaSimple"
forecasting_setting: "time_series"
config: "configs/models/MambaSimple.toml"
registry: "models.mambasimple.registry"
paper_title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
venue: "arXiv preprint"
year: 2023
arxiv: "https://arxiv.org/abs/2312.00752"
---
# MambaSimple

MambaSimple is a time series forecasting model built on the Mamba selective state space architecture. It adapts Mamba's selective scan mechanism — where SSM parameters are functions of the input, allowing the model to selectively propagate or forget information — into a pure PyTorch implementation that requires no custom CUDA operators, making it portable across CPU, CUDA, and MPS backends.

## Paper
- **Title**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Venue**: arXiv preprint
- **Published**: 2023 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2312.00752

## Abstract
Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

## In ModernTSF
Default config: `configs/models/MambaSimple.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2312-00752,
  author       = {Albert Gu and
                  Tri Dao},
  title        = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  journal      = {CoRR},
  volume       = {abs/2312.00752},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2312.00752},
  doi          = {10.48550/ARXIV.2312.00752},
  eprinttype   = {arXiv},
  eprint       = {2312.00752},
  timestamp    = {Sun, 19 Jan 2025 13:42:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2312-00752.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
