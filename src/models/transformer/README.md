---
name: "Transformer"
implementation: upstream
summary: "Transformer is the standard encoder-decoder Transformer architecture applied to long-term time series forecasting. It uses full scaled dot-product self-attention (O(L²) complexity) in both the encoder and decoder, with data embedding (positional + value) on the input. In ModernTSF the upstream TSLib implementation is adapted so that only the long-term forecast path is retained, non-forecasting branches are removed, and shared layer modules (`DataEmbedding`, `FullAttention`, `AttentionLayer`, `Encoder`, `Decoder`) are reused from the shared model utilities."
paper:
  title: "Attention Is All You Need"
  venue: "NeurIPS 2017"
  year: 2017
  url: "https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need"
codebase:
  url: "https://github.com/thuml/Time-Series-Library"
  revision: "2fb5b84ecef67c45a759f7cf82023d27afe27882"
  license: "MIT"
  usage: ported
---
# Transformer

Transformer is the standard encoder-decoder Transformer architecture applied to long-term time series forecasting. It uses full scaled dot-product self-attention (O(L²) complexity) in both the encoder and decoder, with data embedding (positional + value) on the input. In ModernTSF the upstream TSLib implementation is adapted so that only the long-term forecast path is retained, non-forecasting branches are removed, and shared layer modules (`DataEmbedding`, `FullAttention`, `AttentionLayer`, `Encoder`, `Decoder`) are reused from the shared model utilities.

<!-- model-card:canonical:start -->
## Method overview

Transformer is the standard encoder-decoder Transformer architecture applied to long-term time series forecasting.

## Core architecture

It uses full scaled dot-product self-attention (O(L²) complexity) in both the encoder and decoder, with data embedding (positional + value) on the input. In ModernTSF the upstream TSLib implementation is adapted so that only the long-term forecast path is retained, non-forecasting branches are removed, and shared layer modules (`DataEmbedding`, `FullAttention`, `AttentionLayer`, `Encoder`, `Decoder`) are reused from the shared model utilities.

The model-local implementation is in [`model.py`](model.py); imported, strictly
shared building blocks are listed below.

## Input and output

The primary input is a history tensor shaped `[batch, 96, channels]`. The
declared output contract is a `[batch, 96, channels]` point forecast.

## Paper and code

- [paper](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need); title: Attention Is All You Need; venue/year: NeurIPS 2017 / 2017
- [codebase](https://github.com/thuml/Time-Series-Library); revision: `2fb5b84ecef67c45a759f7cf82023d27afe27882`; license: `MIT`; usage: `ported`

## Local implementation

This card declares a `upstream` implementation. Construction and runtime
schema live in [`spec.py`](spec.py), the implementation lives in
[`model.py`](model.py), and the default preset is
[`configs/models/Transformer.toml`](../../../configs/models/Transformer.toml).

## Differences

Implementation: **upstream candidate** (numerical parity is not yet qualified) for the THUML time-series baseline, pinned to
[`thuml/Time-Series-Library`](https://github.com/thuml/Time-Series-Library)
revision `2fb5b84ecef67c45a759f7cf82023d27afe27882` under MIT. The encoder-decoder,
causal decoder attention, cross-attention, feed-forward blocks, positional/time
embedding, and output projection are retained. This is a forecasting integration,
not the original machine-translation pipeline. The ineffective `factor`
setting was removed because full attention never consumes it mathematically.
Direct execution of the pinned source confirms exact output, intermediate,
input-gradient, and every active parameter-gradient parity for the shared
architecture when both sides use the categorical `fixed` embedding. The
default path is nevertheless blocked: ModernTSF passes raw six-column calendar
marks to a six-input `timeF` projection, whereas the pinned upstream hourly
pipeline preprocesses timestamps into four continuous features and its
projection has four inputs. The default state cannot therefore be mapped
completely, so no model-level parity pass is recorded.

## Shared components

- [`embed`](../../components/embed.py)
- [`self_attention_family`](../../components/self_attention_family.py)
- [`transformer_encdec`](../../components/transformer_encdec.py)

## Configuration constraints

The contract fixture uses `seq_len=96` and `pred_len=96`. Default
model parameters are: `enc_in=7`, `d_model=128`, `n_heads=8`, `e_layers=2`, `d_layers=1`, `d_ff=256`, `dropout=0.1`, `activation='gelu'`, `embed='timeF'`, `freq='h'`
<!-- model-card:canonical:end -->

## Paper
- **Title**: Attention Is All You Need
- **Venue**: NeurIPS 2017
- **Published**: 2017 (arXiv: 2017-06)
- **arXiv**: https://arxiv.org/abs/1706.03762

## Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

## In ModernTSF
Default config: `configs/models/Transformer.toml`; model specification: `spec.py`; implementation/adapter: `model.py`.

## Verification

Implementation: **upstream candidate** (numerical parity is not yet qualified) for the THUML time-series baseline, pinned to
[`thuml/Time-Series-Library`](https://github.com/thuml/Time-Series-Library)
revision `2fb5b84ecef67c45a759f7cf82023d27afe27882` under MIT. The encoder-decoder,
causal decoder attention, cross-attention, feed-forward blocks, positional/time
embedding, and output projection are retained. This is a forecasting integration,
not the original machine-translation pipeline. The ineffective `factor`
setting was removed because full attention never consumes it mathematically.
Direct execution of the pinned source confirms exact output, intermediate,
input-gradient, and every active parameter-gradient parity for the shared
architecture when both sides use the categorical `fixed` embedding. The
default path is nevertheless blocked: ModernTSF passes raw six-column calendar
marks to a six-input `timeF` projection, whereas the pinned upstream hourly
pipeline preprocesses timestamps into four continuous features and its
projection has four inputs. The default state cannot therefore be mapped
completely, so no model-level parity pass is recorded.

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
