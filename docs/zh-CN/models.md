# 模型参考

ModernTSF 共内置 38 个模型。每个模型位于 `src/models/<name>/` 目录下，包含三个文件：

- `model.py` — `torch.nn.Module` 实现
- `schema.py` — 用于校验 `model.params` 的 Pydantic `ModelParameterConfig`
- `registry.py` — `register()` 函数，注册模型工厂

模型参数由各模型单独定义，在配置加载时进行校验。具体字段请参考对应的 `schema.py`。

---

## 线性类

简单投影模型，训练速度快，是强有力的基线。

| 名称 | 配置 | 说明 |
|---|---|---|
| `Linear` | `configs/models/Linear.toml` | 按通道对 `seq_len → pred_len` 做线性投影 |
| `DLinear` | `configs/models/DLinear.toml` | 将序列分解为趋势 + 季节性，分别做线性投影 |
| `NLinear` | `configs/models/NLinear.toml` | 先减去最后一个值归一化，再做线性投影 |
| `RLinear` | `configs/models/RLinear.toml` | 带 RevIN（可逆实例归一化）的线性模型 |
| `CrossLinear` | `configs/models/CrossLinear.toml` | 带跨通道交互的线性模型 |
| `MixLinear` | `configs/models/MixLinear.toml` | 时间维与通道维混合线性投影 |

---

## Transformer 类

基于注意力机制的时序依赖建模。

| 名称 | 配置 | 说明 |
|---|---|---|
| `Autoformer` | `configs/models/Autoformer.toml` | 用自相关机制替代自注意力 |
| `FEDformer` | `configs/models/FEDformer.toml` | 频域增强的分解 Transformer |
| `PatchTST` | `configs/models/PatchTST.toml` | 将序列分为 patch，按通道应用 Transformer |
| `iTransformer` | `configs/models/iTransformer.toml` | 倒置 Transformer：对通道做注意力，对时间做 FFN |
| `TimeXer` | `configs/models/TimeXer.toml` | 内生变量分块嵌入 + 外生变量倒置嵌入，通过全局 token 做交叉注意力 |

---

## MLP / Patch 类

前馈与混合架构。

| 名称 | 配置 | 说明 |
|---|---|---|
| `PatchMLP` | `configs/models/PatchMLP.toml` | 基于 patch 的 MLP |
| `xPatch` | `configs/models/xPatch.toml` | 扩展版 patch MLP |
| `TSMixer` | `configs/models/TSMixer.toml` | 时间序列 MLP-Mixer，交替做时间与通道混合 |
| `LightTS` | `configs/models/LightTS.toml` | 轻量级 MLP，基于分块处理 |

---

## CNN 类

| 名称 | 配置 | 说明 |
|---|---|---|
| `TimesNet` | `configs/models/TimesNet.toml` | 将一维时序重塑为二维，应用视觉风格卷积 |
| `SCINet` | `configs/models/SCINet.toml` | 样本卷积与交互网络 |

---

## RNN 类

| 名称 | 配置 | 说明 |
|---|---|---|
| `SegRNN` | `configs/models/SegRNN.toml` | 分段 RNN — 以固定长度分段替代逐步处理 |

---

## 现代预测器

| 名称 | 配置 | 说明 |
|---|---|---|
| `TimeMixer` | `configs/models/TimeMixer.toml` | 多尺度时序混合 |
| `FITS` | `configs/models/FITS.toml` | 频域插值 — 在频域压缩后重建 |
| `SparseTSF` | `configs/models/SparseTSF.toml` | 基于周期对齐采样的稀疏跨周期预测 |
| `CycleNet` | `configs/models/CycleNet.toml` | 从残差中分离周期模式 |
| `TiDE` | `configs/models/TiDE.toml` | 时序稠密编解码器，支持协变量 |

---

## 架构变体

| 名称 | 配置 | 说明 |
|---|---|---|
| `Amplifier` | `configs/models/Amplifier.toml` | 基于放大器的预测器 |
| `TimeBase` | `configs/models/TimeBase.toml` | 时间基础架构 |
| `TimeBridge` | `configs/models/TimeBridge.toml` | 桥接架构 |
| `TimeEmb` | `configs/models/TimeEmb.toml` | 增强时间戳嵌入的模型 |

---

## 滤波类

| 名称 | 配置 | 说明 |
|---|---|---|
| `PaiFilter` | `configs/models/PaiFilter.toml` | 可学习滤波模型 |
| `TexFilter` | `configs/models/TexFilter.toml` | 纹理启发的滤波模型 |

---

## 其他

| 名称 | 配置 | 说明 |
|---|---|---|
| `SVTime` | `configs/models/SVTime.toml` | 基于奇异值分解 |
| `CMoS` | `configs/models/CMoS.toml` | 通道混合结构 |
| `PWS` | `configs/models/PWS.toml` | 分块时序模型 |

---

## 移植的 PoorOtterBob 模型

以下六个模型移植自 [PoorOtterBob](https://github.com/PoorOtterBob) 系列仓库。它们的原始网络结构原样保留（放在 `src/models/<name>/_upstream.py`），并在 `model.py` 中加一层薄适配器。所有模型在此都作为标准时间序列预测器运行，输出 `(B, pred_len, N)`。

适配器通过 `src/models/_external/marks.py` 把 ModernTSF 的 `(x_enc, x_mark_enc, x_dec, x_mark_dec)` 批次转换成各模型原生输入布局：

- **时间序列**模型直接接收数值张量 `(B, T, N)`。
- **时空**模型接收 `(B, T, N, 1 + F)`——数值通道加上 `F = 2` 个归一化日历特征 `[time_in_day, day_in_week]`，沿节点维广播。
- **空气质量**模型还会把未来的日历特征作为解码端协变量输入。

`PHAT` 的上游仓库提供了模型文件，但缺失核心的 `PHAT_Attention` 模块（正负 X 形注意力）。该模块依据论文（ICLR 2026，arXiv:2602.00654 第 3.2 节）在 `src/models/phat/layers/PHAT_Attention.py` 中复现，文件内记录了公式到代码的对应关系；PHAT 其余部分原样移植。

> ⚠️ **未验证的复现**：`PHAT_Attention` 因作者从未公开而依据论文重建。它能以正确的张量形状前向与反向传播，但**无法验证**是否忠实于作者的真实实现。在用作者代码验证之前，请把 `PHAT` 的结果当作尽力而为的近似，**而非论文数值的复现**。

| 名称 | 配置 | 类别 | 说明 |
|---|---|---|---|
| `MoFo` | `configs/models/MoFo.toml` | 时间序列 | 周期模式 Transformer，周期对齐 patch |
| `PHAT` | `configs/models/PHAT.toml` | 时间序列 | 周期异质性 Transformer；`PHAT_Attention` ⚠️ **未验证**的论文重建（arXiv:2602.00654），非论文复现 |
| `BiST` | `configs/models/BiST.toml` | 时空 | 轻量双向 MLP，自适应图 |
| `MAGE` | `configs/models/MAGE.toml` | 时空 | 自适应图专家混合 |
| `STOP` | `configs/models/STOP.toml` | 时空 | 解耦基座 MLP + Core_Adaptive 残差校正 |
| `CauAir` | `configs/models/CauAir.toml` | 空气质量 | 因果协变量注意力，使用未来协变量 |
| `AirCade` | `configs/models/AirCade.toml` | 空气质量 | 因果解耦，使用未来协变量，默认 `freq_mae` 损失 |

`AirCade` 要求 `pred_len == seq_len`（其时间长度固定），默认使用频域 MAE 损失（`loss = "freq_mae"`）；`MoFo` 的 `freq_weighted_mae` 也可选。每个模型的端到端冒烟运行配置在 `configs/runs/smoke_*.toml`——先用 `python scripts/make_smoke_data.py` 生成合成数据。

---

## 共享模块

可复用的构建模块位于 `src/models/module/`：

| 模块 | 内容 |
|---|---|
| `embed.py` | 位置编码、时间特征嵌入、patch 嵌入 |
| `self_attention_family.py` | 点积、加性、Autoformer、FEDformer 注意力变体 |
| `fourier_correlation.py` | 频域互相关 |
| `auto_correlation.py` | 自相关计算 |
| `positional_encoding.py` | 正弦位置编码 |
| `revin.py` | RevIN — 可逆实例归一化 |
| `masking.py` | 三角因果掩码 |
| `conv_blocks.py` | 卷积构建块 |
| `transformer_encdec.py` | 标准 Transformer 编解码器层 |
| `autoformer_encdec.py` | Autoformer 专用编解码器 |
| `tst_transformer.py` | PatchTST Transformer 层 |
| `standard_norm.py` | InstanceNorm 封装 |

---

## 模型接口

所有模型遵循统一接口：

```python
# 构造器接收解包后的 model.params
model = Model(c_in=7, seq_len=512, pred_len=96, **other_params)

# forward 签名 — 不使用的参数用 *args 接收
def forward(self, x, x_mark, dec_inp, dec_mark):
    ...
```

`registry.py` 中注册的工厂接收 `(cfg: RootConfig, params: dict)`：

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
