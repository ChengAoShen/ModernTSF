---
model: "MAGE"
forecasting_setting: "spatiotemporal"
config: "configs/models/MAGE.toml"
registry: "models.mage.registry"
paper_title: "Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting"
venue: "NeurIPS 2025"
year: 2025
arxiv: ""
---
# MAGE

MAGE (Mixture of Adaptive Graph Experts) is a spatiotemporal learning model for node-structured or graph-structured data. It introduces a sparse yet balanced mixture-of-experts strategy in which each expert perceives a unique underlying graph topology through kernel-based functions with linear complexity relative to the number of nodes, overcoming the noise amplification caused by ReLU activations in existing adaptive graph learning methods.

## Paper
- **Title**: Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting
- **Venue**: NeurIPS 2025
- **Published**: 2025
- **arXiv**: N/A

## Abstract
The effectiveness of Spatiotemporal Graph Neural Networks (STGNNs) critically hinges on the quality of the underlying graph topology. While end-to-end adaptive graph learning methods have demonstrated promising results in capturing latent spatiotemporal dependencies, they often suffer from high computational complexity and limited expressive capacity. In this paper, we propose MAGE for efficient spatiotemporal forecasting. We first conduct a theoretical analysis demonstrating that the ReLU activation function employed in existing methods amplifies edge-level noise during graph topology learning, thereby compromising the fidelity of the learned graph structures. To enhance model expressiveness, we introduce a sparse yet balanced mixture-of-experts strategy, where each expert perceives the unique underlying graph through kernel-based functions and operates with linear complexity relative to the number of nodes. The sparsity mechanism ensures that each node interacts exclusively with compatible experts, while the balancing mechanism promotes uniform activation across all experts, enabling diverse and adaptive graph representations. Furthermore, we theoretically establish that a single graph convolution using the learned graph in MAGE is mathematically equivalent to multiple convolutional steps under conventional graphs. We evaluate MAGE against advanced baselines on multiple real-world spatiotemporal datasets. MAGE achieves competitive performance while maintaining strong computational efficiency.

## In ModernTSF
Default config: `configs/models/MAGE.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{ma2025less,
  author    = {Jiaming Ma and Binwu Wang and Guanjun Wang and Kuo Yang and Zhengyang Zhou and Pengkun Wang and Xu Wang and Yang Wang},
  title     = {Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MAGE}
}
```
