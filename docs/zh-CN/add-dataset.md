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
