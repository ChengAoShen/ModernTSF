---
model: "WPMixer"
forecasting_setting: "time_series"
config: "configs/models/WPMixer.toml"
registry: "models.wpmixer.registry"
paper_title: "WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting"
venue: "AAAI 2025"
year: 2025
arxiv: "https://arxiv.org/abs/2412.17176"
---
# WPMixer

WPMixer (Wavelet Patch Mixer) is an MLP-based model for long-term time series forecasting in the standard time series setting. It combines three complementary techniques: multi-resolution wavelet decomposition to extract information in both frequency and time domains, patching to capture extended historical context and local patterns with an extended look-back window, and MLP mixing layers to incorporate global temporal information — significantly outperforming state-of-the-art MLP-based and Transformer-based models in a computationally efficient manner.

## Paper
- **Title**: WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting
- **Venue**: AAAI 2025
- **Published**: 2025 (arXiv: 2024-12)
- **arXiv**: https://arxiv.org/abs/2412.17176

## Abstract
Time series forecasting is crucial for various applications, such as weather forecasting, power load forecasting, and financial analysis. In recent studies, MLP-mixer models for time series forecasting have been shown as a promising alternative to transformer-based models. However, the performance of these models is still yet to reach its potential. In this paper, we propose Wavelet Patch Mixer (WPMixer), a novel MLP-based model, for long-term time series forecasting, which leverages the benefits of patching, multi-resolution wavelet decomposition, and mixing. Our model is based on three key components: (i) multi-resolution wavelet decomposition, (ii) patching and embedding, and (iii) MLP mixing. Multi-resolution wavelet decomposition efficiently extracts information in both the frequency and time domains. Patching allows the model to capture an extended history with a look-back window and enhances capturing local information while MLP mixing incorporates global information. Our model significantly outperforms state-of-the-art MLP-based and transformer-based models for long-term time series forecasting in a computationally efficient way, demonstrating its efficacy and potential for practical applications.

## In ModernTSF
Default config: `configs/models/WPMixer.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.
