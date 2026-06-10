---
model: "Transformer"
forecasting_setting: "time_series"
config: "configs/models/Transformer.toml"
registry: "models.transformer.registry"
paper_title: "Attention Is All You Need"
venue: "NeurIPS 2017"
year: 2017
arxiv: "https://arxiv.org/abs/1706.03762"
---
# Transformer

Transformer is the standard encoder-decoder Transformer architecture applied to long-term time series forecasting. It uses full scaled dot-product self-attention (O(L²) complexity) in both the encoder and decoder, with data embedding (positional + value) on the input. In ModernTSF the upstream TSLib implementation is adapted so that only the long-term forecast path is retained, non-forecasting branches are removed, and shared layer modules (`DataEmbedding`, `FullAttention`, `AttentionLayer`, `Encoder`, `Decoder`) are reused from the shared model utilities.

## Paper
- **Title**: Attention Is All You Need
- **Venue**: NeurIPS 2017
- **Published**: 2017 (arXiv: 2017-06)
- **arXiv**: https://arxiv.org/abs/1706.03762

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## In ModernTSF
Default config: `configs/models/Transformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@misc{vaswani2017attention,
  author        = {Ashish Vaswani and
                  Noam Shazeer and
                  Niki Parmar and
                  Jakob Uszkoreit and
                  Llion Jones and
                  Aidan N. Gomez and
                  Lukasz Kaiser and
                  Illia Polosukhin},
  title         = {Attention Is All You Need},
  year          = {2017},
  eprint        = {1706.03762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1706.03762}
}
```
