# 概率预测（`output_type`）

点预测 vs. 概率预测是一个独立的**输出**轴，与 `task.mode` 正交。`task.mode`
（`time_series` / `spatiotemporal` / `covariate`，参见
[task-modes.md](task-modes.md)）选择的是*数据设定*——一个 batch 的形状以及模型
接收到什么。`output_type` 选择的是模型 `forward` *返回*什么——单一的点预测，
还是带有校准不确定性的预测。两者可以自由组合：概率模型原则上可以用于任意
`task.mode`，不过目前已实现的每个概率模型（`QuantileDLinear`、
`QuantilePatchTST`、`MQRNN`、`TiRex`、`GaussianMLP`、`GaussianProcessTS`、
`DeepAR`）都面向 `time_series`。

流水线通过 `getattr(model, "output_type", "point")` 读取这个轴，只有当它不是
`"point"` 时才走额外分支。点模型从不设置这个属性，因此走的是不变的默认路径——
加入概率支持没有改动约 170 个既有点模型。

## 三种 `output_type` 取值

| `output_type` | `forward` 返回 | 含义 |
|---|---|---|
| `"point"`（默认） | `(B, pred_len, C)` | 每个时间步/通道一个值。 |
| `"quantile"` | `(B, pred_len, C, Q)` | `Q = len(quantile_levels)` 个分位数组成的网格，沿最后一维升序且不交叉，顺序与 `quantile_levels` 一致。 |
| `"distribution"` | `(B, pred_len, C, 2)` | 某分布的 `(loc, scale)`（当前 `distribution_family = "gaussian"`；`scale > 0`）。 |

`C = c_out = 1 if features == "MS" else enc_in`——与点模型相同的通道约定。概率
模型在构建末尾的 `Q`/`2` 维之前，需先切片到目标通道。

### 单调的 `QuantileHead`

分位数模型不应手写分位数头——它们应包裹共享的 `QuantileHead`
（`src/models/_quantile_head.py`）。给定每步的基础特征张量
`(B, L, C, in_features)`，它先投影出一个中位数锚点（`anchor_proj`），再为每个
分位数间隙投影出严格非负的偏移量（`softplus(offset_proj(base))`），然后向上
用累加和构建中位数以上的分位数，向下用累加和构建中位数以下的分位数。由于每个
间隙都 `>= 0`，输出**在构造上就不会交叉**，与学到的权重无关——输出
`[..., m]`（最接近 `0.5` 的分位水平）等于锚点，且数组沿最后一维单调不减。

`quantile_levels` 必须升序，否则 `QuantileHead.__init__` 会报错，因为损失函数
和指标都按位置索引这个末尾维度为 `levels[i]`。

使用 `QuantileHead` 的模板：`src/models/quantile_dlinear/`（最小实现，约 40
行，包裹 DLinear）、`src/models/quantile_patchtst/`（包裹 Transformer 骨干）、
`src/models/mqrnn/` 和 `src/models/tirex/`（从零实现的 RNN / TiRex 适配器）。

## 损失函数配对

每种 `output_type` 都由匹配的损失函数训练，注册在
`src/benchmark/losses_prob.py` 中，通过 `[training] loss = "..."` 选择（点
损失 `mse`/`mae`/`l1` 不变，仍是默认值）：

| `output_type` | `[training] loss` | 损失模块 | 公式 |
|---|---|---|---|
| `"quantile"` | `"quantile"` | `QuantileLoss` | 加权 pinball 损失：对分位水平 `q`，`max(q * (y - yhat_q), (q - 1) * (y - yhat_q))`，在 batch、horizon、通道和分位水平上取平均。 |
| `"distribution"` | `"nll_gaussian"` | `GaussianNLLLoss` | 高斯负对数似然：`0.5 * log(2*pi*scale^2) + 0.5 * ((y - loc) / scale)^2`，取均值；`scale` 被截断为 `>= eps`（默认 `1e-6`）。 |

两个损失函数都接收 rank-4 的预测张量，以及 trainer 已经通过
`_slice_pred_target` 切好的普通 rank-3 目标 `(B, pred_len, C)`。

### `quantile_levels`：唯一真源

`evaluation.quantile_levels`（`src/benchmark/config/schema/evaluation.py`）
是分位水平的规范列表——默认是九个十分位数
`[0.1, 0.2, ..., 0.9]`。`run_one` 将它注入两个地方：

- **模型构建**：如果模型 `__init__` 的签名中含有 `quantile_levels` 参数
  （通过 `inspect.signature` 检测），且配置中尚未设置
  `model.params.quantile_levels`，`run_one` 会自动注入
  `list(config.evaluation.quantile_levels)`，从而保证 `QuantileHead` 的 `Q`
  始终与配置的分位水平一致。
- **`training.loss_params`**：当 `loss == "quantile"` 且 `loss_params` 中尚未
  存在 `quantile_levels` 时，`run_one` 会在构造损失函数前，将其注入到
  `loss_params` 的一份*拷贝*中。

点模型永远不会看到这个字段；除非模型或损失函数主动使用它，否则
`evaluation.quantile_levels` 会被忽略。

## 概率预测运行的评估指标

`collect_prob_metrics`（`src/benchmark/evaluation/metrics.py`）为任意非点
模型的运行计算恰好四个指标，输入是原始预测张量、rank-3 目标、配置的
`levels` 以及模型的 `output_type`：

| 指标 | 方向 | 计算方式 |
|---|---|---|
| `crps` | 越小越好 | 当 `output_type == "distribution"` 且 `distribution_family == "gaussian"` 时用高斯 CRPS 的闭式解；否则用基于分位数网格的近似 `(2/Q) * sum_q mean(pinball_q)`。 |
| `wql` | 越小越好 | GIFT-Eval/GluonTS 风格的加权分位数损失：`(1/Q) * sum_q ( 2 * sum(pinball_q) / sum|y| )`。对分位数路径而言，该值等于 `crps / mean|y|`。 |
| `coverage_80` | 诊断性指标——越接近 `0.8` 越好，不强制排序方向 | 真实值落在 `[q_0.1, q_0.9]`（由 `0.1`/`0.9` 分位水平确定的中心 80% 区间；若配置中没有 `0.1`/`0.9`，则回退到配置的最低/最高分位水平）内的比例。 |
| `width_80` | 越小越好（其他条件相同时） | `[q_0.1, q_0.9]` 区间的平均宽度。 |

对于 `"distribution"` 模型，`collect_prob_metrics` 会先从 `(loc, scale)` 在
配置的 `levels` 上构建一个内部分位数网格，再计算
`wql`/`coverage_80`/`width_80`（`crps` 单独使用高斯闭式公式）。

概率指标与常规的点指标（`mae`/`mse`/`rmse`/...）**同时**运行，后者对
`"quantile"` 模型计算在**中位数**分位数上，对 `"distribution"` 模型计算在
**`loc`** 上，因此概率模型与点模型仍可在同一榜单上比较。
`crps`/`wql`/`coverage_80`/`width_80` 必须显式列在 `[evaluation] metrics`
中才能通过该次运行的指标白名单——它们不在默认指标列表中。

## 示例配置

模型配置只声明架构参数——`output_type` 和损失函数配对分别在模型的
`model.py` 和运行配置中体现。

`configs/models/QuantileDLinear.toml`（分位数）：

```toml
[model]
name = "QuantileDLinear"

[model.params]
enc_in = 7
kernel_size = 25
individual = false
```

`configs/models/GaussianMLP.toml`（分布）：

```toml
[model]
name = "GaussianMLP"

[model.params]
enc_in = 7
hidden_size = 256
num_layers = 2
dropout = 0.1
```

运行/smoke 配置在任意此类模型配置之上接入损失函数与指标（模式见
`probabilistic-forecasting` 技能，沿用 `configs/runs/smoke_crib.toml` 相同的
`extends` 写法）：

```toml
extends = ["../base.toml", "../datasets/smoke.toml", "../models/QuantileDLinear.toml"]

[task]
seq_len = 96
pred_len = 12
features = "M"

[training]
loss = "quantile"          # 分布模型用 "nll_gaussian"

[evaluation]
metrics = ["crps", "wql", "coverage_80", "width_80", "mae", "mse"]
# quantile_levels 默认为 9 个十分位数；如需可在此覆盖
```

其他已实现的概率模型遵循相同的两种配对：`QuantilePatchTST` 和 `TiRex`
（`output_type = "quantile"`，损失 `"quantile"`），`GaussianProcessTS` 和
`DeepAR`（`output_type = "distribution"`，损失 `"nll_gaussian"`）。

## 添加新的概率模型

使用 **`probabilistic-forecasting`** 技能，而不是普通的 `add-model` 流程
（后者构建的是点模型）。该技能会引导完成：

- 声明 `self.output_type = "quantile"`（用 `QuantileHead` 包裹骨干网络）或
  `self.output_type = "distribution"` + `self.distribution_family =
  "gaussian"`（用 `scale = softplus(...) + eps` 输出 `(loc, scale)`）。
- 在 `Model.__init__` 和模型的 `schema.py` 中都声明
  `quantile_levels: list[float] | None = None` 参数，以便上文所述的
  `run_one` 自动注入生效。
- 像其他任意模型一样接入——`registry.py`、
  `src/benchmark/registry/models.py` 中的 `MODEL_NAME_MAP` 条目、
  `configs/models/<Name>.toml`，以及一个 smoke 配置——基础流程参见
  [add-model.md](add-model.md)。
- 选择匹配的损失函数（`"quantile"` / `"nll_gaussian"`），并在
  `[evaluation] metrics` 中列出四个概率指标名称。
- 用 `uv run python tool/tsf.py smoke --model <Name>` 验证：PASS 意味着该
  次运行能够训练并输出有限的 `crps`/`wql`/`coverage_80`/`width_80`（以及
  点指标），并满足合理性检查 `wql < ~1`、`width_80 > 0`、
  `coverage_80 in [0, 1]`。

关键文件：

| 内容 | 位置 |
|---|---|
| 单调分位数头 | `src/models/_quantile_head.py` |
| 概率损失函数 | `src/benchmark/losses_prob.py` |
| 概率评估指标 | `src/benchmark/evaluation/metrics.py`（`collect_prob_metrics`） |
| `output_type` 分支逻辑 | `src/benchmark/runner/{trainer,evaluator,run_one}.py` |
| `quantile_levels` 配置 | `src/benchmark/config/schema/evaluation.py` |
| 模板 | `src/models/{quantile_dlinear,quantile_patchtst,mqrnn,tirex,gaussian_mlp,deepar,gaussian_process_ts}/` |
