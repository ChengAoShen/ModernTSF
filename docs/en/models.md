# Models reference

ModernTSF includes 38 models. Each model lives under `src/models/<name>/` and has three files:

- `model.py` — `torch.nn.Module` implementation
- `schema.py` — Pydantic `ModelParameterConfig` for validating `model.params`
- `registry.py` — `register()` function that registers the model factory

Model parameters are defined per-model and validated at config load time. See the corresponding `schema.py` for the exact fields.

---

## Linear-based

Simple projection models. Fast to train and strong baselines.

| Name key | Config | Notes |
|---|---|---|
| `Linear` | `configs/models/Linear.toml` | Per-channel linear projection over `seq_len → pred_len` |
| `DLinear` | `configs/models/DLinear.toml` | Decomposes series into trend + seasonal, applies linear to each |
| `NLinear` | `configs/models/NLinear.toml` | Normalises by subtracting the last value before linear projection |
| `RLinear` | `configs/models/RLinear.toml` | Linear with RevIN (reversible instance normalisation) |
| `CrossLinear` | `configs/models/CrossLinear.toml` | Linear with cross-channel interaction |
| `MixLinear` | `configs/models/MixLinear.toml` | Mixed temporal and channel linear projections |

---

## Transformer-based

Attention-based models for temporal dependency modelling.

| Name key | Config | Notes |
|---|---|---|
| `Autoformer` | `configs/models/Autoformer.toml` | Auto-correlation mechanism replaces self-attention |
| `FEDformer` | `configs/models/FEDformer.toml` | Frequency-enhanced decomposed transformer |
| `PatchTST` | `configs/models/PatchTST.toml` | Divides series into patches, applies transformer per channel |
| `iTransformer` | `configs/models/iTransformer.toml` | Inverted transformer: attention over channels, FFN over time |
| `TimeXer` | `configs/models/TimeXer.toml` | Patch endogenous + inverted exogenous embedding with global-token cross-attention |
| `Informer` | `configs/models/Informer.toml` | ProbSparse self-attention with distilling for efficient long-sequence forecasting |
| `Crossformer` | `configs/models/Crossformer.toml` | Cross-dimension attention over patched segments via a two-stage attention router |
| `Transformer` | `configs/models/Transformer.toml` | Vanilla encoder-decoder transformer with full dot-product self-attention |
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

---

## MLP / Patch-based

Feed-forward and mixing architectures.

| Name key | Config | Notes |
|---|---|---|
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

---

## CNN-based

| Name key | Config | Notes |
|---|---|---|
| `TimesNet` | `configs/models/TimesNet.toml` | Reshapes 1D time series to 2D, applies vision-style convolution |
| `SCINet` | `configs/models/SCINet.toml` | Sample convolution and interaction network |
| `MICN` | `configs/models/MICN.toml` | Multi-scale isometric convolution capturing local + global temporal patterns |
| `ModernTCN` | `configs/models/ModernTCN.toml` | Modernised temporal convolutional network with large-kernel depthwise convolutions |
| `WaveNet` | `configs/models/WaveNet.toml` | Stacked dilated causal convolutions with gated activations and residual/skip connections |

---

## RNN-based

| Name key | Config | Notes |
|---|---|---|
| `SegRNN` | `configs/models/SegRNN.toml` | Segmented RNN — processes fixed-length segments instead of step-by-step |
| `DeepAR` | `configs/models/DeepAR.toml` | Autoregressive recurrent network producing probabilistic forecasts |

---

## Modern forecasters

| Name key | Config | Notes |
|---|---|---|
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

---

## Architecture variants

| Name key | Config | Notes |
|---|---|---|
| `Amplifier` | `configs/models/Amplifier.toml` | Amplifier-based forecaster |
| `TimeBase` | `configs/models/TimeBase.toml` | Time-based architecture |
| `TimeBridge` | `configs/models/TimeBridge.toml` | Bridging architecture |
| `TimeEmb` | `configs/models/TimeEmb.toml` | Enhanced with time-stamp embeddings |

---

## Filter-based

| Name key | Config | Notes |
|---|---|---|
| `PaiFilter` | `configs/models/PaiFilter.toml` | Learnable filter-based model |
| `TexFilter` | `configs/models/TexFilter.toml` | Texture-inspired filtering |

---

## Other

| Name key | Config | Notes |
|---|---|---|
| `SVTime` | `configs/models/SVTime.toml` | Singular-value based decomposition |
| `CMoS` | `configs/models/CMoS.toml` | Channel mixing structure |
| `PWS` | `configs/models/PWS.toml` | Patch-wise series model |
| `Sumba` | `configs/models/Sumba.toml` | Dynamic graph-convolution forecaster with dilated-inception temporal blocks |

---

## Ported PoorOtterBob models

These six models are ported from the [PoorOtterBob](https://github.com/PoorOtterBob)
repositories. They keep their original architecture verbatim (vendored under
`src/models/<name>/_upstream.py`) and add a thin benchmark-facing adapter in
`model.py`. All run here as standard time-series forecasters returning
`(B, pred_len, N)`.

The adapter converts ModernTSF's `(x_enc, x_mark_enc, x_dec, x_mark_dec)` batch
into each model's native input layout via `src/models/_external/marks.py`:

- **Time series** models receive the value tensor `(B, T, N)` directly.
- **Spatiotemporal** models receive `(B, T, N, 1 + F)` — the value channel plus
  `F = 2` normalized calendar features `[time_in_day, day_in_week]` broadcast
  across nodes.
- **Air-quality** models additionally consume the future calendar features as
  decoder-side covariates.

`PHAT`'s upstream repository ships its model file but omits the core
`PHAT_Attention` module (the Positive-Negative X-shape Attention). It is
reconstructed from the paper (ICLR 2026, arXiv:2602.00654, Section 3.2) in
`src/models/phat/layers/PHAT_Attention.py`, with the equation-to-code mapping
documented in that file; the rest of PHAT is vendored verbatim.

> ⚠️ **Unverified reconstruction.** The `PHAT_Attention` module was rebuilt
> from the paper because the authors never released it. It runs and
> back-propagates with correct tensor shapes, but its fidelity to the authors'
> actual implementation **cannot be verified**. Treat `PHAT` results as a
> best-effort approximation, **not** a reproduction of the paper's numbers,
> until validated against the authors' own code.

| Name key | Config | Category | Notes |
|---|---|---|---|
| `MoFo` | `configs/models/MoFo.toml` | Time series | Periodic-pattern transformer; period-aligned patches |
| `PHAT` | `configs/models/PHAT.toml` | Time series | Period-heterogeneity transformer; `PHAT_Attention` ⚠️ **unverified** reconstruction from the paper (arXiv:2602.00654) — not a paper reproduction |
| `BiST` | `configs/models/BiST.toml` | Spatiotemporal | Lightweight bidirectional MLP with adaptive graph |
| `MAGE` | `configs/models/MAGE.toml` | Spatiotemporal | Mixture of adaptive-graph experts |
| `STOP` | `configs/models/STOP.toml` | Spatiotemporal | Decoupled base MLP + Core_Adaptive residual correction |
| `CauAir` | `configs/models/CauAir.toml` | Air quality | Causal covariate attention; uses future covariates |
| `AirCade` | `configs/models/AirCade.toml` | Air quality | Causal decoupling; future covariates; trains with `freq_mae` |

`AirCade` requires `pred_len == seq_len` (its temporal length is fixed) and
defaults to the frequency-domain MAE loss (`loss = "freq_mae"`); `MoFo`'s
`freq_weighted_mae` is also available. A tiny end-to-end smoke run for each model
lives in `configs/runs/smoke_*.toml` — generate the synthetic data first with
`python scripts/make_smoke_data.py`.

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
