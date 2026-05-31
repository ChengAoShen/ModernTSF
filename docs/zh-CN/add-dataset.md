# 如何加入新数据集

数据集通过 `DATASET_NAME_MAP` 与模块级 `register()` 注册，并通过 schema 校验 `dataset.params`。

根据数据来源不同，支持两种模式：

---

## 模式 A：单文件数据集（读取时自动切分）

适用于拥有单个 CSV 文件、需要 ModernTSF 自动切分训练/验证/测试集的情况。

### 1) 实现数据集

在 `src/data/datasets/` 下新增：

```text
src/data/datasets/my_dataset.py
```

继承 `ForecastingDataset` 并实现 `_read_data`。

在 `_read_data` 中通常需要：

- 读取原始数据（CSV/文本/合成）。
- 处理 `features`（`M`/`MS`/`S`）。
- 根据 `scale` 进行缩放。
- 使用 `_get_borders` 根据 `split_ratio` 切分。
- 返回 `(series_data, time_stamp)` 两个 `np.ndarray`。

```python
class Dataset_Custom(ForecastingDataset):
    def _read_data(self, flag, features, target, split_ratio, scale):
        df_raw = pd.read_csv(self.file_path)
        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)
        # ... 特征选择与缩放 ...
        return series_data, time_stamp
```

### 2) 编写参数 schema

在 `src/data/schemas/datasets/` 新增：

```python
from pydantic import BaseModel, Field


class DatasetParameterConfig(BaseModel):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])
```

### 3) 注册数据集

在数据集文件中添加 `register()`：

```python
from benchmark.registry import DATASET_REGISTRY
from data.schemas.datasets.my_dataset import DatasetParameterConfig


def register() -> None:
    DATASET_REGISTRY.register("my_dataset", Dataset_My, DatasetParameterConfig)
```

### 4) 更新 DATASET_NAME_MAP

编辑 `src/benchmark/registry/datasets.py`：

```python
DATASET_NAME_MAP["my_dataset"] = "data.datasets.my_dataset"
```

### 5) 添加数据集配置

新增 `configs/datasets/my_dataset.toml`：

```toml
[dataset]
name = "my_dataset"
root_path = "./dataset/my_dataset"
data_path = "my.csv"

[dataset.params]
target = "OT"
scale = true
split_ratio = [0.7, 0.1, 0.2]
```

### 6) 在入口配置中使用

```toml
extends = ["../../base.toml", "../../datasets/my_dataset.toml", "../../models/DLinear.toml"]
```

### 快捷方式：普通 CSV 用 `name = "custom"`

如果数据是单个扁平多变量 CSV（一个 `date` 列加若干数值通道），则**无需**第
1–4 步。在配置中设置 `name = "custom"` 即可复用内置 `Dataset_Custom` 加载器——
只写配置，无需新代码：

```toml
[dataset]
name = "custom"
root_path = "./dataset/exchange_rate"
data_path = "exchange_rate.csv"

[dataset.params]
target = "OT"
scale = true
split_ratio = [0.7, 0.1, 0.2]
```

已内置示例：`exchange`、`ili`、`beijing_air`、`aqshunyi`、`aqwan`、`nn5`、
`fred_md`。

---

## 模式 B：预切分数据集（文件夹内含 train/val/test）

适用于数据已提前切分为独立文件的情况。框架内置的 `presplit` 数据集无需编写任何代码即可支持。

### 文件夹结构

```text
dataset/my_dataset/
  train.csv
  val.csv
  test.csv
```

三个文件须有相同的列结构。`date` 列可选——若存在则生成时间特征，否则使用零时间戳。

### 数据集配置

```toml
[dataset]
name = "presplit"
root_path = "./dataset/my_dataset"
data_path = ""

[dataset.params]
target = "OT"
scale = true
```

scaler 始终在 `train.csv` 上拟合，保证验证/测试集使用一致的归一化参数。

### 在入口配置中使用

```toml
extends = ["../../base.toml", "../../datasets/my_dataset.toml", "../../models/DLinear.toml"]
```

无需编写自定义数据集类或 schema。

---

## 备注

- CSV 数据集推荐包含 `date` 列用于时间特征（模式 A 需要；模式 B 视为可选）。
- 使用模式 A 的合成数据集可忽略 `data_path`，直接在 `_read_data` 中生成序列。
- `features = "S"` 时使用 `target` 选择输出通道。

---

## 模式 C：节点结构化数据集（spatiotemporal / covariate）

当 `task.mode = "spatiotemporal"` 或 `"covariate"` 时使用本模式：`N` 个节点中每个携带一个数值加 `F` 个逐节点协变量。这类数据集返回标准四元组契约，**数值**放序列槽、**协变量**放时间戳槽：

```
__getitem__ -> (value_hist (T,N), value_fut (T,N), cov_hist (T,N,F), cov_fut (T,N,F))
```

`cov_fut` 是空气质量模型消费的未来协变量块。节点结构化数据集是普通的 `torch.utils.data.Dataset`（无需继承 `ForecastingDataset`）；建议设类属性 `spatiotemporal = True`，并实现 `__len__` / `__getitem__` / `inverse_transform`。

两个内置示例：

- `synthetic_st`（`src/data/datasets/synthetic_st.py`）——生成带日历协变量 `[time_in_day, day_in_week]` 的小型 `(T, N, 3)` 张量。
- `cauair_st` / `cauair_ts`（`src/data/datasets/cauair.py`）——加载 CauAir 的索引窗口 `.npz` 包（`data (T, N, C)`、`idx_{train,val,test}.npy`、可选 `adj_mx.npy`）。`cauair_st` 暴露节点布局用于 spatiotemporal / covariate 模式；`cauair_ts` 把 `N` 个节点数值摊平为 `C` 个通道用于普通预测。

```toml
[dataset]
name = "cauair_st"
root_path = "./dataset/cauair_ccaq"
data_path = ""

[dataset.params]
input_dim = 8      # 数值 + (input_dim - 1) 个协变量
npz_name = "his.npz"
scale = true
```

交通图数据包（`metr_la`、`pems_bay`、`pems03/04/07/08`）复用同一个 `cauair_st`
加载器——没有专门的交通数据集。用 `tool/convert_traffic.py` 把原始数值矩阵 + 邻接
转换成 bundle。如何转换原始数组以及各配置指向何处见
`docs/zh-CN/datasets-traffic.md`。

当图模型消费 bundle 中的 `adj_mx.npy` 时，可在 `[dataset.params]` 下设置 `adj_norm`
（如 `adj_norm = "gcn"`）让 runner 在注入前对邻接做归一化；未设置时原样使用原始矩阵。
方案列表见 `docs/zh-CN/task-modes.md`。

各模式如何塑造批次、以及模型兼容性见 `docs/zh-CN/task-modes.md`。
