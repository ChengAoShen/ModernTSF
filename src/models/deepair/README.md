---
model: "DeepAir"
forecasting_setting: "covariate"
config: "configs/models/DeepAir.toml"
registry: "models.deepair.registry"
paper_title: "Deep Distributed Fusion Network for Air Quality Prediction"
venue: "KDD 2018"
year: 2018
arxiv: ""
---
# DeepAir

DeepAir is a covariate prediction model for air quality forecasting that combines a spatial transformation component — which converts sparse air quality observations into a consistent input representation — with a deep distributed fusion network that integrates heterogeneous urban data (air quality, meteorology, weather forecasts) to predict 48-hour-ahead air quality for multiple monitoring stations.

## Paper
- **Title**: Deep Distributed Fusion Network for Air Quality Prediction
- **Venue**: KDD 2018
- **Published**: 2018
- **arXiv**: N/A

## Abstract
Accompanying the rapid urbanization, many developing countries are suffering from serious air pollution problem. The demand for predicting future air quality is becoming increasingly more important to government's policy-making and people's decision making. In this paper, we predict the air quality of next 48 hours for each monitoring station, considering air quality data, meteorology data, and weather forecast data. Based on the domain knowledge about air pollution, we propose a deep neural network (DNN)-based approach (entitled DeepAir), which consists of a spatial transformation component and a deep distributed fusion network. Considering air pollutants' spatial correlations, the former component converts the spatial sparse air quality data into a consistent input to simulate the pollutant sources. The latter network adopts a neural distributed architecture to fuse heterogeneous urban data for simultaneously capturing the factors affecting air quality, e.g. meteorological conditions. We deployed DeepAir in our AirPollutionPrediction system, providing fine-grained air quality forecasts for 300+ Chinese cities every hour. The experimental results on the data from three-year nine Chinese-city demonstrate the advantages of DeepAir beyond 10 baseline methods. Comparing with the previous online approach in AirPollutionPrediction system, we have 2.4%, 12.2%, 63.2% relative accuracy improvements on short-term, long-term and sudden changes prediction, respectively.

## In ModernTSF
Default config: `configs/models/DeepAir.toml`; parameter schema: `schema.py`; implementation/adapter: `model.py`; registry entry: `registry.py`.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/YiZWLZ18,
  author       = {Xiuwen Yi and
                  Junbo Zhang and
                  Zhaoyuan Wang and
                  Tianrui Li and
                  Yu Zheng},
  editor       = {Yike Guo and
                  Faisal Farooq},
  title        = {Deep Distributed Fusion Network for Air Quality Prediction},
  booktitle    = {Proceedings of the 24th {ACM} {SIGKDD} International Conference on
                  Knowledge Discovery {\&} Data Mining, {KDD} 2018, London, UK, August
                  19-23, 2018},
  pages        = {965--973},
  publisher    = {{ACM}},
  year         = {2018},
  url          = {https://doi.org/10.1145/3219819.3219822},
  doi          = {10.1145/3219819.3219822},
  timestamp    = {Sun, 02 Nov 2025 21:27:16 +0100},
  biburl       = {https://dblp.org/rec/conf/kdd/YiZWLZ18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
