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
