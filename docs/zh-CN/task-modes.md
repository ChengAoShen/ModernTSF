# 任务模式

ModernTSF 的所有任务都是**预测（forecasting）**；`task.mode` 选择的是*数据设定*——即一个 batch 的张量形状以及模型接收到的内容。默认是 `time_series`，因此已有配置不受影响。

```toml
[task]
mode = "time_series"   # 或 "spatiotemporal" | "covariate"
seq_len = 96
label_len = 0
pred_len = 96
```

## `time_series`（默认）

经典多变量时间序列预测。一个 batch 是数值张量 `(B, T, C)`，每个通道既是输入也是目标。日历特征是从数据集 `date` 列构造的原始时间戳 `(B, T, 6)` = `[year, month, day, weekday, hour, minute]`（无日期列时补零）。这是 ModernTSF 原有行为——已有模型与数据集完全不变。

## `spatiotemporal`（时空）

节点结构化预测。`N` 个节点中每个都携带一个数值加 `F` 个逐节点协变量，所以自然输入是 `(B, T, N, 1 + F)`。ModernTSF 通过标准的四元组数据集契约承载它：**数值**放在序列槽 `(B, T, N)`，**协变量**放在时间戳槽 `(B, T, N, F)`：

```
__getitem__ -> (value_hist (T,N), value_fut (T,N), cov_hist (T,N,F), cov_fut (T,N,F))
```

时空模型从数值与协变量标记重建 `(B, T, N, 1 + F)`。唯一目标是所有 `N` 个节点的数值通道，输出为 `(B, pred_len, N)`。

协变量有两种形态：

- **日历协变量**（`F = 2`，`[time_in_day, day_in_week]` 取值 `[0, 1)`）。`BiST`、`MAGE`、`STOP` 把它们当作嵌入索引使用。`synthetic_st` 数据集产生这种布局。
- **任意协变量**（任意 `F`）。`CauAir` / `AirCade` 用线性层投影协变量，因此气象类协变量可直接使用。

## `covariate`（协变量）

与 `spatiotemporal` 类似，但模型还会接收**未来（已知）**协变量块——预测窗口上的协变量——通过未来时间戳 `(B, pred_len, N, F)`。这是已知未来外生变量、但不知未来目标值的预测模型所使用的解码端协变量输入，例如空气质量模型（`CauAir`、`AirCade`）：它们已知未来气象但不知未来污染物数值。在这些模型上设置 `cov_dim = F`，使未来协变量块尺寸正确。

## 模型 / 模式 兼容性

| 模型 | time_series | spatiotemporal | covariate |
|---|:---:|:---:|:---:|
| `MoFo`、`PHAT` 及内置预测模型 | ✓ | | |
| `BiST`、`MAGE`、`STOP` | ✓（日历标记） | ✓（日历协变量） | |
| `CauAir`、`AirCade` | ✓ | ✓ | ✓（未来协变量） |

模型适配器是多态的：3 维标记 `(B, T, 6)` 被当作原始日历时间戳（time_series），4 维标记 `(B, T, N, F)` 被当作节点结构化协变量（spatiotemporal / covariate）。详见 `src/models/_external/marks.py`。

## 各模式的数据集

- `time_series` — 任意 CSV 数据集（ETT、weather、custom……）。
- `spatiotemporal` — `synthetic_st`（日历协变量）或 `cauair_st`（CauAir / CCAQ 气象）。
- `covariate` — `cauair_st`（提供未来协变量块）。
- 同一份 CauAir 数据也可作为普通时间序列数据集 `cauair_ts`，此时 `N` 个节点数值成为 `C` 个通道。

端到端最小冒烟运行见 `configs/runs/smoke_st_bist.toml`（时空）与 `configs/runs/smoke_cov_cauair.toml`（协变量）。

## 图模型与邻接矩阵

图模型（如 STGCN、DCRNN、GraphWaveNet）需要 `(N, N)` 邻接矩阵。节点结构化数据集会把它暴露为 `self.adj_mx`（从 bundle 里的 `adj_mx.npy` 加载，无则为 `None`）。Runner 从 **train** 数据集读取并注入到模型工厂，因此图模型的 `registry.py` 可直接取用：

```python
lambda cfg, params: Model(
    seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
    num_nodes=params["num_nodes"],          # 由数据集注入
    adj_mx=params.get("adj_mx"),            # (N, N) np.ndarray 或 None
    ...
)
```

`params["adj_mx"]` 与 `params["num_nodes"]` 在 schema 校验**之后**注入（见 `src/benchmark/runner/run_one.py`），所以无需写进模型 TOML——它们来自数据。非图模型自动忽略。

### 可选的邻接归一化（`adj_norm`）

在 `[dataset.params]` 下设置 `adj_norm`，可在注入前对来自数据的邻接矩阵做归一化。未设置时原始矩阵原样传入，因此自带归一化的图模型不受影响。支持的方案（`src/models/_external/adj_norm.py`）：

| `adj_norm` | 函数 |
|---|---|
| `sym_norm_lap` / `symmetric_normalized_laplacian` | 对称归一化拉普拉斯 |
| `scaled_laplacian` | 缩放拉普拉斯（Chebyshev） |
| `gcn` / `gcn_norm` | GCN 重归一化（`D^-½ (A+I) D^-½`） |
| `transition` / `transition_matrix` | 随机游走转移矩阵 |
| `reverse_transition` / `reverse_transition_matrix` | 反向随机游走转移矩阵 |

```toml
[dataset.params]
adj_norm = "gcn"
```

要用真实交通数据集（METR-LA、PEMS-BAY、PEMS0x），用 `tool/convert_traffic.py` 把原始数值矩阵 + 邻接转换成节点 bundle，再让 `cauair_st` 数据集配置指向输出目录：

```bash
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr-la.npz --values-key data \
    --adj dataset/metr_la/adj_mx.npy \
    --output-dir dataset/metr_la --add-time --freq-min 5
```

## 范围：仅预测

三种模式都是预测任务。其他任务类型——**插补**、**异常检测**、**分类**、
**基础模型预训练**——刻意不在范围内：它们各自需要不同的数据格式、任务契约
和评估协议。代码库中没有 `task_name` 参数或任何非预测分支；部分上游
TSLib 风格模型自带的多任务分支在移植时已被剥离。
