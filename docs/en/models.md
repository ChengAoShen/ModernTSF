# Models reference

ModernTSF includes 31 models. Each model lives under `src/models/<name>/` and has three files:

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

---

## MLP / Patch-based

Feed-forward and mixing architectures.

| Name key | Config | Notes |
|---|---|---|
| `PatchMLP` | `configs/models/PatchMLP.toml` | Patch-based MLP |
| `xPatch` | `configs/models/xPatch.toml` | Extended patch-based model |
| `TSMixer` | `configs/models/TSMixer.toml` | MLP-Mixer for time series (alternates time and channel mixing) |
| `LightTS` | `configs/models/LightTS.toml` | Lightweight MLP with chunk-based processing |

---

## CNN-based

| Name key | Config | Notes |
|---|---|---|
| `TimesNet` | `configs/models/TimesNet.toml` | Reshapes 1D time series to 2D, applies vision-style convolution |
| `SCINet` | `configs/models/SCINet.toml` | Sample convolution and interaction network |

---

## RNN-based

| Name key | Config | Notes |
|---|---|---|
| `SegRNN` | `configs/models/SegRNN.toml` | Segmented RNN — processes fixed-length segments instead of step-by-step |

---

## Modern forecasters

| Name key | Config | Notes |
|---|---|---|
| `TimeMixer` | `configs/models/TimeMixer.toml` | Multi-scale time series mixing |
| `FITS` | `configs/models/FITS.toml` | Frequency interpolation — compresses and reconstructs in frequency domain |
| `SparseTSF` | `configs/models/SparseTSF.toml` | Sparse cross-period forecasting with period-aligned sampling |
| `CycleNet` | `configs/models/CycleNet.toml` | Separates recurrent cycle patterns from residuals |
| `TiDE` | `configs/models/TiDE.toml` | Time-series dense encoder-decoder with covariate support |

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
