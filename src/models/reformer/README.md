---
model: "Reformer"
forecasting_setting: "time_series"
config: "configs/models/Reformer.toml"
registry: "models.reformer.registry"
paper_title: "Reformer: The Efficient Transformer"
venue: "ICLR 2020"
year: 2020
arxiv: "https://arxiv.org/abs/2001.04451"
---
# Reformer

Reformer is a memory-efficient Transformer model adapted for the time-series forecasting setting. It replaces standard dot-product self-attention with locality-sensitive hashing (LSH) attention, reducing the attention complexity from O(L²) to O(L log L), and employs reversible residual layers to avoid storing all intermediate activations, making it practical for long input sequences.

## Paper
- **Title**: Reformer: The Efficient Transformer
- **Venue**: ICLR 2020
- **Published**: 2020
- **arXiv**: https://arxiv.org/abs/2001.04451

## Abstract
Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L²) to O(L log L), where L is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.

## In ModernTSF
Default config: `configs/models/Reformer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
