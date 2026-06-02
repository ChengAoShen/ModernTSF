# Models reference

ModernTSF includes 172 models. Each model lives under `src/models/<name>/` and has a local `README.md` with YAML front matter, plus the implementation files used by the runner:

- `model.py` — `torch.nn.Module` implementation or adapter
- `schema.py` — Pydantic `ModelParameterConfig` for validating `model.params`
- `registry.py` — `register()` function that registers the model factory

Model parameters are defined per model and validated at config load time. See the corresponding `schema.py` for exact fields.

The catalogue is grouped by forecasting setting, not by architecture family.

---

## Time Series

Ordinary univariate or multivariate forecasting with `(B, T, C)` history tensors.
These models cover linear baselines, Transformers, MLP / patch models, CNNs,
RNNs, state-space models, filtering methods, recent 2025/2026 conference models,
and other architecture variants.

| Name key | Config | Notes |
|---|---|---|
| `Linear` | `configs/models/Linear.toml` | Per-channel linear projection over `seq_len → pred_len` |
| `DLinear` | `configs/models/DLinear.toml` | Decomposes series into trend + seasonal, applies linear to each |
| `NLinear` | `configs/models/NLinear.toml` | Normalises by subtracting the last value before linear projection |
| `RLinear` | `configs/models/RLinear.toml` | Linear with RevIN (reversible instance normalisation) |
| `CrossLinear` | `configs/models/CrossLinear.toml` | Linear with cross-channel interaction |
| `MixLinear` | `configs/models/MixLinear.toml` | Mixed temporal and channel linear projections |
| `PatchTST` | `configs/models/PatchTST.toml` | Divides series into patches, applies transformer per channel |
| `iTransformer` | `configs/models/iTransformer.toml` | Inverted transformer: attention over channels, FFN over time |
| `TimeXer` | `configs/models/TimeXer.toml` | Patch endogenous + inverted exogenous embedding with global-token cross-attention |
| `Crossformer` | `configs/models/Crossformer.toml` | Cross-dimension attention over patched segments via a two-stage attention router |
| `Informer` | `configs/models/Informer.toml` | ProbSparse self-attention with distilling for efficient long-sequence forecasting |
| `Autoformer` | `configs/models/Autoformer.toml` | Auto-correlation mechanism replaces self-attention |
| `FEDformer` | `configs/models/FEDformer.toml` | Frequency-enhanced decomposed transformer |
| `Reformer` | `configs/models/Reformer.toml` | Efficient transformer using LSH attention to reduce memory and compute |
| `Pyraformer` | `configs/models/Pyraformer.toml` | Pyramidal attention over a multi-resolution tree for long-range dependencies |
| `ETSformer` | `configs/models/ETSformer.toml` | Exponential-smoothing attention with level/growth/season decomposition |
| `NSTransformer` | `configs/models/NSTransformer.toml` | Non-stationary transformer with de-stationary attention and series stationarization |
| `MultiPatchFormer` | `configs/models/MultiPatchFormer.toml` | Multi-scale patch embedding with cross-patch transformer attention |
| `PAttn` | `configs/models/PAttn.toml` | Patch embedding fed straight into a single self-attention block — a minimalist patch transformer baseline |
| `CARD` | `configs/models/CARD.toml` | Channel-aligned robust dual-attention transformer mixing token and channel attention |
| `Fredformer` | `configs/models/Fredformer.toml` | Frequency-debiased transformer attending over per-frequency patches to counter low-frequency bias |
| `DUET` | `configs/models/DUET.toml` | Dual clustering on temporal and channel dimensions with a fusion module |
| `Pathformer` | `configs/models/Pathformer.toml` | Multi-scale transformer with adaptive pathways routing patches across temporal resolutions |
| `DSFormer` | `configs/models/DSFormer.toml` | Double-sampling transformer with TVA (temporal-variable attention) encoder/decoder blocks |
| `DTAF` | `configs/models/DTAF.toml` | Patch-embedding transformer with decomposition stabilization and frequency-differencing wave modeling |
| `TimePerceiver` | `configs/models/TimePerceiver.toml` | Perceiver-style architecture: iterative cross/self attention over patches with query-based future decoding |
| `Transformer` | `configs/models/Transformer.toml` | Vanilla encoder-decoder transformer with full dot-product self-attention |
| `PatchMLP` | `configs/models/PatchMLP.toml` | Patch-based MLP |
| `xPatch` | `configs/models/xPatch.toml` | Extended patch-based model |
| `TSMixer` | `configs/models/TSMixer.toml` | MLP-Mixer for time series (alternates time and channel mixing) |
| `LightTS` | `configs/models/LightTS.toml` | Lightweight MLP with chunk-based processing |
| `WPMixer` | `configs/models/WPMixer.toml` | Wavelet-patch MLP-mixer over multi-level decomposed sub-series |
| `MTSMixer` | `configs/models/MTSMixer.toml` | Factorized MLP-mixer disentangling temporal and channel interactions for multivariate forecasting |
| `UMixer` | `configs/models/UMixer.toml` | U-Net-style multi-scale mixing with a stationarity-correction module |
| `NHiTS` | `configs/models/NHiTS.toml` | Neural hierarchical interpolation: multi-rate sampling + hierarchical interpolation MLP stacks |
| `NBeats` | `configs/models/NBeats.toml` | Deep stack of fully-connected basis-expansion blocks with backcast/forecast residuals |
| `HDMixer` | `configs/models/HDMixer.toml` | Hierarchical patch mixer with length-extendable patches for multivariate forecasting |
| `SRSNet` | `configs/models/SRSNet.toml` | Selective representation space: dual patch views (selective + dynamic) with an MLP forecast head |
| `TimesNet` | `configs/models/TimesNet.toml` | Reshapes 1D time series to 2D, applies vision-style convolution |
| `SCINet` | `configs/models/SCINet.toml` | Sample convolution and interaction network |
| `MICN` | `configs/models/MICN.toml` | Multi-scale isometric convolution capturing local + global temporal patterns |
| `ModernTCN` | `configs/models/ModernTCN.toml` | Modernised temporal convolutional network with large-kernel depthwise convolutions |
| `WaveNet` | `configs/models/WaveNet.toml` | Stacked dilated causal convolutions with gated activations and residual/skip connections |
| `SegRNN` | `configs/models/SegRNN.toml` | Segmented RNN — processes fixed-length segments instead of step-by-step |
| `DeepAR` | `configs/models/DeepAR.toml` | Autoregressive recurrent network producing probabilistic forecasts |
| `MambaSimple` | `configs/models/MambaSimple.toml` | Selective state-space (Mamba) sequence model — dependency-free pure-PyTorch selective scan, no CUDA kernels required |
| `S_Mamba` | `configs/models/S_Mamba.toml` | iTransformer-style inverted embedding with a Mamba block over the channel dimension; kernel-free selective scan |
| `BiMamba` | `configs/models/BiMamba.toml` | Bidirectional Mamba scanning the sequence forward and backward; kernel-free selective scan |
| `S4` | `configs/models/S4.toml` | Structured state-space (S4D diagonal) sequence model with frequency-domain convolution kernels |
| `TimeMixer` | `configs/models/TimeMixer.toml` | Multi-scale time series mixing |
| `FITS` | `configs/models/FITS.toml` | Frequency interpolation — compresses and reconstructs in frequency domain |
| `SparseTSF` | `configs/models/SparseTSF.toml` | Sparse cross-period forecasting with period-aligned sampling |
| `CycleNet` | `configs/models/CycleNet.toml` | Separates recurrent cycle patterns from residuals |
| `TiDE` | `configs/models/TiDE.toml` | Time-series dense encoder-decoder with covariate support |
| `FiLM` | `configs/models/FiLM.toml` | Frequency-improved Legendre memory with low-rank approximation |
| `FreTS` | `configs/models/FreTS.toml` | Frequency-domain MLPs over real/imaginary spectral components |
| `Koopa` | `configs/models/Koopa.toml` | Koopman-theory operator separating time-invariant and time-variant dynamics |
| `SOFTS` | `configs/models/SOFTS.toml` | Series-core fusion with a STar Aggregate-Redistribute module for channel interaction |
| `TimeKAN` | `configs/models/TimeKAN.toml` | Kolmogorov-Arnold network with multi-scale frequency decomposition for forecasting |
| `Amplifier` | `configs/models/Amplifier.toml` | Amplifier-based forecaster |
| `TimeBase` | `configs/models/TimeBase.toml` | Time-based architecture |
| `TimeBridge` | `configs/models/TimeBridge.toml` | Bridging architecture |
| `TimeEmb` | `configs/models/TimeEmb.toml` | Enhanced with time-stamp embeddings |
| `PaiFilter` | `configs/models/PaiFilter.toml` | Learnable filter-based model |
| `TexFilter` | `configs/models/TexFilter.toml` | Texture-inspired filtering |
| `SVTime` | `configs/models/SVTime.toml` | Singular-value based decomposition |
| `CMoS` | `configs/models/CMoS.toml` | Channel mixing structure |
| `PWS` | `configs/models/PWS.toml` | Patch-wise series model |
| `Sumba` | `configs/models/Sumba.toml` | Dynamic graph-convolution forecaster with dilated-inception temporal blocks |
| `CrossGNN` | `configs/models/CrossGNN.toml` | Cross-scale and cross-variable graph network modeling multi-scale interactions without an external adjacency |
| `MSGNet` | `configs/models/MSGNet.toml` | Multi-scale inter-series graph network — FFT-selected periods with an internal adaptive variate graph (no external adjacency) |
| `TimeFilter` | `configs/models/TimeFilter.toml` | Patch-specific spatial-temporal graph filtration learning an internal patch graph (no external adjacency) |
| `MoFo` | `configs/models/MoFo.toml` | Periodic-pattern transformer; period-aligned patches |
| `PHAT` | `configs/models/PHAT.toml` | Period-heterogeneity transformer; `PHAT_Attention` ⚠️ **unverified** reconstruction from the paper (arXiv:2602.00654) — not a paper reproduction |
| `CATS` | `configs/models/CATS.toml` | Query-adaptive masking transformer with cross-attention to future tokens |
| `RidgeRegressionTS` | `configs/models/RidgeRegressionTS.toml` | Torch-native ridge-regression style lag forecaster with L2 regularization |
| `LassoRegressionTS` | `configs/models/LassoRegressionTS.toml` | Torch-native Lasso-style lag forecaster with L1 regularization |
| `ElasticNetTS` | `configs/models/ElasticNetTS.toml` | Elastic-Net style lag forecaster combining L1 and L2 penalties |
| `BayesianRidgeTS` | `configs/models/BayesianRidgeTS.toml` | Bayesian-ridge inspired linear forecaster with shrinkage regularization |
| `PolynomialRegressionTS` | `configs/models/PolynomialRegressionTS.toml` | Polynomial lag forecaster over raw, squared, and square-root history features |
| `KNNForecasterTS` | `configs/models/KNNForecasterTS.toml` | Differentiable KNN-style prototype forecaster with RBF weights |
| `SVRForecasterTS` | `configs/models/SVRForecasterTS.toml` | Support-vector-regression inspired RBF prototype forecaster with a linear skip |
| `GaussianProcessTS` | `configs/models/GaussianProcessTS.toml` | Gaussian-process inspired prototype-kernel forecaster |
| `DecisionTreeTS` | `configs/models/DecisionTreeTS.toml` | Single differentiable soft decision tree over lag features |
| `RandomForestTS` | `configs/models/RandomForestTS.toml` | Random-forest style soft-tree ensemble |
| `ExtraTreesTS` | `configs/models/ExtraTreesTS.toml` | Extra-Trees style randomized shallow soft-tree ensemble |
| `GradientBoostingTS` | `configs/models/GradientBoostingTS.toml` | Gradient-boosting style residual soft-tree ensemble |
| `XGBoostTS` | `configs/models/XGBoostTS.toml` | XGBoost-style residual soft-tree ensemble registered as a Torch forecaster |
| `LightGBMTS` | `configs/models/LightGBMTS.toml` | LightGBM-style lightweight residual soft-tree ensemble |
| `CatBoostTS` | `configs/models/CatBoostTS.toml` | CatBoost-style ordered-residual soft-tree ensemble |
| `ARIMATS` | `configs/models/ARIMATS.toml` | ARIMA-inspired differentiable forecaster over historical differences |
| `AutoRegressiveTS` | `configs/models/AutoRegressiveTS.toml` | Autoregressive lag-window forecaster |
| `ExpSmoothingTS` | `configs/models/ExpSmoothingTS.toml` | Exponential-smoothing inspired forecaster with learnable decay and trend extrapolation |
| `KalmanFilterTS` | `configs/models/KalmanFilterTS.toml` | Kalman-filter inspired alpha-beta smoother with learnable update gains |
| `MLPForecasterTS` | `configs/models/MLPForecasterTS.toml` | Basic MLP lag-window forecaster with channel mixing |
| `RNNForecasterTS` | `configs/models/RNNForecasterTS.toml` | Basic vanilla-RNN sequence forecaster |
| `GRUForecasterTS` | `configs/models/GRUForecasterTS.toml` | Basic GRU sequence forecaster |
| `LSTMForecasterTS` | `configs/models/LSTMForecasterTS.toml` | Basic LSTM sequence forecaster under the time-series setting |
| `TCNForecasterTS` | `configs/models/TCNForecasterTS.toml` | Small temporal convolutional forecaster |
| `Aurora` | `configs/models/Aurora.toml` | Universal multimodal time-series foundation-model adapter with phase, spectral, and channel context. |
| `TimeAlign` | `configs/models/TimeAlign.toml` | Distribution-aware alignment forecaster that matches horizon statistics to the recent context. |
| `GTR` | `configs/models/GTR.toml` | Global temporal retrieval adapter for mixing local windows with long-cycle temporal context. |
| `PhaseFormer` | `configs/models/PhaseFormer.toml` | Phase-domain forecaster that aggregates period-aligned historical patterns. |
| `PMDformer` | `configs/models/PMDformer.toml` | Patch-mean decoupling forecaster that separates local shape from trend level. |
| `MMPD` | `configs/models/MMPD.toml` | Multi-mode patch diffusion inspired adapter for diverse time-series forecasts. |
| `COSA` | `configs/models/COSA.toml` | Context-aware output-space adaptation forecaster for test-time forecast correction. |
| `DistDF` | `configs/models/DistDF.toml` | Joint-distribution alignment adapter inspired by Wasserstein forecast-label matching. |
| `Sonnet` | `configs/models/Sonnet.toml` | Spectral-operator neural forecaster emphasizing smooth harmonic components. |
| `APN` | `configs/models/APN.toml` | Adaptive periodic network style forecaster with phase projection. |
| `TimeCAP` | `configs/models/TimeCAP.toml` | Channel-aware pretraining inspired adapter with context-aware temporal prompts. |
| `GOTSF` | `configs/models/GOTSF.toml` | Goal-oriented forecaster that can bias predictions toward application-specific target ranges. |
| `FTP` | `configs/models/FTP.toml` | FusionTimePatch-style adapter joining channel-independent and channel-mixed temporal views. |
| `OccamVTS` | `configs/models/OccamVTS.toml` | Vision-model distillation inspired forecaster represented as multimodal temporal gating. |
| `HN_MVTS` | `configs/models/HN_MVTS.toml` | Hypernetwork-style hierarchical adapter for multivariate time-series forecasting. |
| `SEMPO` | `configs/models/SEMPO.toml` | Lightweight foundation-model adapter with spectral decomposition and prompt-expert routing. |
| `InterPDN` | `configs/models/InterPDN.toml` | Per-step probabilistic distribution modeling adapter using stabilized ordinal horizons. |
| `TimeO1` | `configs/models/TimeO1.toml` | Transformed-label alignment inspired adapter for post-decoding forecast correction. |
| `FeTS` | `configs/models/FeTS.toml` | Feature-aware forecasting adapter that learns sparse temporal importance masks. |
| `SymTime` | `configs/models/SymTime.toml` | Symbolic time-series foundation-model adapter constrained around recent level and scale. |
| `ImplicitForecaster` | `configs/models/ImplicitForecaster.toml` | Implicit neural decoder that forms forecasts from latent time coordinates. |
| `AMRC` | `configs/models/AMRC.toml` | Adaptive masking-loss adapter with representation-consistency inspired temporal core retention. |
| `HMformer` | `configs/models/HMformer.toml` | Hierarchical multi-scale Transformer style adapter for long-term forecasting. |
| `TiRex` | `configs/models/TiRex.toml` | Zero-shot xLSTM-inspired forecasting adapter represented as a temporal expert portfolio. |
| `LatentTSF` | `configs/models/LatentTSF.toml` | Latent-state forecasting adapter that decodes future values from compact hidden states. |
| `CoRA` | `configs/models/CoRA.toml` | Correlation-aware adapter for multivariate forecasting foundation models. |
| `DynamicTMoE` | `configs/models/DynamicTMoE.toml` | Drift-aware dynamic mixture-of-experts adapter for non-stationary forecasting. |
| `PULSE` | `configs/models/PULSE.toml` | Generative phase-evolution adapter for non-stationary time-series forecasting. |
| `OLinear` | `configs/models/OLinear.toml` | Orthogonally transformed linear forecasting adapter with normalized channel mixing. |
| `MAFS` | `configs/models/MAFS.toml` | Multi-agent forecasting adapter that combines specialized temporal experts. |
| `TSRAG` | `configs/models/TSRAG.toml` | Retrieval-augmented time-series foundation-model adapter for zero-shot forecasting. |
| `TimeMosaic` | `configs/models/TimeMosaic.toml` | Adaptive-granularity patch and segment decoding adapter for heterogeneous time series. |
| `Kronos` | `configs/models/Kronos.toml` | Large-scale time-series foundation-model adapter with prompt-style temporal conditioning. |

---

## Spatiotemporal Learning

Node-structured or graph forecasting models that learn temporal dynamics together
with spatial or node relationships. These models consume value histories plus
node/calendar covariates through the `spatiotemporal` data setting.

| Name key | Config | Notes |
|---|---|---|
| `BiST` | `configs/models/BiST.toml` | Lightweight bidirectional MLP with adaptive graph |
| `MAGE` | `configs/models/MAGE.toml` | Mixture of adaptive-graph experts |
| `STOP` | `configs/models/STOP.toml` | Decoupled base MLP + Core_Adaptive residual correction |
| `STID` | `configs/models/STID.toml` | Spatial-temporal identity MLP with node/time-of-day/day-of-week embeddings |
| `GWNet` | `configs/models/GWNet.toml` | Graph WaveNet: adaptive adjacency + dilated causal convolutions |
| `STGCN` | `configs/models/STGCN.toml` | Spatio-temporal graph convolutional network (graph + temporal conv blocks) |
| `DCRNN` | `configs/models/DCRNN.toml` | Diffusion-convolutional recurrent network (dual random-walk graph conv in a GRU) |
| `MTGNN` | `configs/models/MTGNN.toml` | Learns the graph structure jointly with mix-hop graph + dilated temporal convolution |
| `AGCRN` | `configs/models/AGCRN.toml` | Adaptive graph conv GRU with node-adaptive parameters (learns adjacency from node embeddings) |
| `STNorm` | `configs/models/STNorm.toml` | Spatial + temporal normalization on a WaveNet backbone (graph-free) |
| `StemGNN` | `configs/models/StemGNN.toml` | Spectral-temporal GNN (graph + discrete Fourier transforms) with a learned latent graph |
| `STGODE` | `configs/models/STGODE.toml` | Graph neural ODE for continuous spatiotemporal dynamics |
| `STAEformer` | `configs/models/STAEformer.toml` | Spatio-temporal adaptive embedding transformer (attention over time and nodes) |
| `GTS` | `configs/models/GTS.toml` | Learns a discrete graph structure jointly with a DCRNN-style recurrent forecaster |
| `DGCRN` | `configs/models/DGCRN.toml` | Dynamic graph convolutional recurrent network (time-varying adjacency in a GRU) |
| `STDN` | `configs/models/STDN.toml` | Spatio-temporal decoupled network |
| `DFDGCN` | `configs/models/DFDGCN.toml` | Data-driven frequency dynamic graph convolution network (vendored from GestaltCogTeam/DFDGCN, MIT) |
| `STPGNN` | `configs/models/STPGNN.toml` | Spatio-temporal pivotal graph neural network |
| `D2STGNN` | `configs/models/D2STGNN.toml` | Decoupled dynamic spatial-temporal graph network (separates diffusion and inherent signals with a dynamic graph) |
| `MegaCRN` | `configs/models/MegaCRN.toml` | Meta-graph convolutional recurrent network with a memory-augmented graph learner |
| `HimNet` | `configs/models/HimNet.toml` | Hierarchical interaction memory network for spatiotemporal forecasting |
| `BigST` | `configs/models/BigST.toml` | Linear-complexity spatiotemporal GNN scaling to large graphs via random-feature linear attention |
| `STWave` | `configs/models/STWave.toml` | Disentangled trend/event spatiotemporal transformer using discrete wavelet decomposition |
| `STTN` | `configs/models/STTN.toml` | Spatial-temporal transformer network (decoupled spatial + temporal attention) |
| `DSTAGNN` | `configs/models/DSTAGNN.toml` | Dynamic spatial-temporal aware GNN (data-driven dynamic graph + multi-head attention) |
| `HL` | `configs/models/HL.toml` | Historical Last — repeats the last observed step (naive baseline) |
| `LSTM` | `configs/models/LSTM.toml` | Plain per-node LSTM sequence forecaster |
| `RPMixer` | `configs/models/RPMixer.toml` | Random-projection MLP-mixer |

---

## Covariate Prediction

The original air-quality forecasting family. These models target node values and
use historical covariates; models with decoder-side covariate blocks also use
known future covariates through the `covariate` data setting.

| Name key | Config | Notes |
|---|---|---|
| `CauAir` | `configs/models/CauAir.toml` | Causal covariate attention; uses future covariates |
| `AirCade` | `configs/models/AirCade.toml` | Causal decoupling; future covariates; trains with `freq_mae` |
| `ASTGCN` | `configs/models/ASTGCN.toml` | Attention-based spatial-temporal GCN (spatial + temporal attention over Chebyshev graph convolution) |
| `GCLSTM` | `configs/models/GCLSTM.toml` | Graph-convolutional LSTM (Chebyshev graph conv inside the LSTM gates) |
| `DeepAir` | `configs/models/DeepAir.toml` | Fusion-based deep air-quality forecaster |
| `GAGNN` | `configs/models/GAGNN.toml` | Group-aware graph neural network (group/city-level attention plus a GNN) |
| `PM25_GNN` | `configs/models/PM25_GNN.toml` | GNN + GRU PM2.5 forecaster with domain-knowledge edges |
| `AirFormer` | `configs/models/AirFormer.toml` | Causal temporal attention with stochastic latent variables for air quality |
| `PCDCNet` | `configs/models/PCDCNet.toml` | Physics/causal-guided dynamic convolution network |
| `AirPhyNet` | `configs/models/AirPhyNet.toml` | Physics-informed network with diffusion/advection ODEs (needs `torchdiffeq`) |
| `AirDualODE` | `configs/models/AirDualODE.toml` | Dual ODE system (physics + data-driven) with knowledge fusion (needs `torchdiffeq`) |
| `MGSFformer` | `configs/models/MGSFformer.toml` | Multi-granularity spatial-temporal fusion transformer |

---

## Shared modules

Reusable building blocks live in `src/models/module/`:

| Module | Contents |
|---|---|
| `embed.py` | Positional encoding, time feature embeddings, patch embeddings |
| `self_attention_family.py` | Dot-product, additive, Autoformer, FEDformer attention variants |
| `fourier_correlation.py` | Frequency-domain cross-correlation |
| `auto_correlation.py` | Auto-correlation computation |
| `positional_encoding.py` | Sinusoidal positional encoding |
| `revin.py` | RevIN — reversible instance normalisation |
| `masking.py` | Triangular causal mask |
| `conv_blocks.py` | Convolutional building blocks |
| `transformer_encdec.py` | Standard transformer encoder / decoder layers |
| `autoformer_encdec.py` | Autoformer-specific encoder / decoder |
| `tst_transformer.py` | PatchTST transformer layers |
| `standard_norm.py` | InstanceNorm wrapper |

---

## Model interface

All models follow the same interface:

```python
# Constructor receives unpacked model.params
model = Model(c_in=7, seq_len=512, pred_len=96, **other_params)

# Forward signature — unused args should be accepted with *args
def forward(self, x, x_mark, dec_inp, dec_mark):
    ...
```

The factory registered in `registry.py` receives `(cfg: RootConfig, params: dict)`:

```python
def register() -> None:
    MODEL_REGISTRY.register(
        "MyModel",
        lambda cfg, params: Model(
            c_in=cfg.dataset.params.get("enc_in", 7),
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            **params,
        ),
        ModelParameterConfig,
    )
```
