---
model: "S_Mamba"
forecasting_setting: "time_series"
config: "configs/models/S_Mamba.toml"
registry: "models.s_mamba.registry"
paper_title: "Is Mamba Effective for Time Series Forecasting?"
venue: "arXiv preprint"
year: 2024
arxiv: "https://arxiv.org/abs/2403.11144"
---
# S_Mamba

S_Mamba (Simple-Mamba) is a time series forecasting model that applies selective state space modeling in an iTransformer-style inverted embedding scheme. It tokenizes each variate's time points via a linear layer, uses a bidirectional Mamba layer to extract inter-variate correlations across the channel dimension, and applies a feed-forward network to learn temporal dependencies, finally mapping to forecasts through a linear layer without requiring custom CUDA operators for selective scanning.

## Paper
- **Title**: Is Mamba Effective for Time Series Forecasting?
- **Venue**: arXiv preprint
- **Published**: 2024
- **arXiv**: https://arxiv.org/abs/2403.11144

## Abstract
In the realm of time series forecasting (TSF), it is imperative for models to adeptly discern and distill hidden patterns within historical time series data to forecast future states. Transformer-based models exhibit formidable efficacy in TSF, primarily attributed to their advantage in apprehending these patterns. However, the quadratic complexity of the Transformer leads to low computational efficiency and high costs, which somewhat hinders the deployment of the TSF model in real-world scenarios. Recently, Mamba, a selective state space model, has gained traction due to its ability to process dependencies in sequences while maintaining near-linear complexity. For TSF tasks, these characteristics enable Mamba to comprehend hidden patterns as the Transformer and reduce computational overhead compared to the Transformer. Therefore, we propose a Mamba-based model named Simple-Mamba (S-Mamba) for TSF. Specifically, we tokenize the time points of each variate autonomously via a linear layer. A bidirectional Mamba layer is utilized to extract inter-variate correlations and a Feed-Forward Network is set to learn temporal dependencies. Finally, the generation of forecast outcomes through a linear mapping layer. Experiments on thirteen public datasets prove that S-Mamba maintains low computational overhead and achieves leading performance. Furthermore, we conduct extensive experiments to explore Mamba's potential in TSF tasks.

## In ModernTSF
Default config: `configs/models/S_Mamba.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@article{DBLP:journals/ijon/WangKFWYZWZ25,
  author       = {Zihan Wang and
                  Fanheng Kong and
                  Shi Feng and
                  Ming Wang and
                  Xiaocui Yang and
                  Han Zhao and
                  Daling Wang and
                  Yifei Zhang},
  title        = {Is Mamba effective for time series forecasting?},
  journal      = {Neurocomputing},
  volume       = {619},
  pages        = {129178},
  year         = {2025},
  url          = {https://doi.org/10.1016/j.neucom.2024.129178},
  doi          = {10.1016/J.NEUCOM.2024.129178},
  timestamp    = {Sat, 15 Nov 2025 13:50:19 +0100},
  biburl       = {https://dblp.org/rec/journals/ijon/WangKFWYZWZ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
