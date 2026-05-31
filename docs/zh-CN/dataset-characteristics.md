# 数据集特征分析

ModernTSF 提供了一个独立的、受 TFB 启发的分析层，用于量化数据集的统计特性。
它纯粹用于诊断：不参与训练，也不引入任何新依赖（仅使用 numpy/scipy）。

`tool/dataset_characteristics.py`

## 基本用法

```bash
uv run python tool/dataset_characteristics.py --config configs/datasets/etth1.toml --split train
```

该脚本以与 `tool/visual_data.py` 相同的方式加载数据集分片（复用数据集注册表），
计算各项特征，打印表格，并写出 CSV。使用 `--per-channel` 可额外输出每个通道的行。

## 主要参数

- `--config`：TOML 配置文件路径。可以是完整的运行配置或仅含数据集的配置。
- `--split`：`train`、`val` 或 `test`（默认 `train`）。
- `--period`：季节周期。未设置时，从主导 FFT 频率自动估计。
- `--per-channel`：额外输出每个通道一行（否则仅输出数据集级别一行）。
- `--out`：输出 CSV 路径。默认 `work_dirs/<dataset>/characteristics_<split>.csv`。

## 计算的特征

| 列 | 含义 |
|---|---|
| `period` | 用于 STL 风格分解的季节周期（FFT 估计或 `--period`）。 |
| `trend_strength` | STL 风格 `1 - Var(resid) / Var(resid + trend)`，使用移动平均趋势。取值 `[0, 1]`。 |
| `seasonality_strength` | STL 风格 `1 - Var(resid) / Var(resid + seasonal)`，使用按周期平均的季节分量。取值 `[0, 1]`。 |
| `stationarity` | 越高越平稳。安装了 `statsmodels` 时为 ADF 检验的 `1 - p 值`，否则为轻量级滚动均值/方差稳定性比率。`stationarity_method` 列记录所用方法。 |
| `shifting` | 前后两半段的均值漂移绝对值，按序列标准差归一化。 |
| `transition` | 一阶（lag-1）自相关。 |
| `correlation` | 各通道两两相关系数绝对值的均值（仅数据集行；每通道行为 `n/a`）。 |

数据集级别的行对各通道的单变量统计量取均值（跨通道平均），
但 `correlation` 是基于全部通道一起计算的。

## 关于平稳性

`statsmodels` **不是**本项目的依赖。当它缺失时（默认情况），`stationarity` 列
回退为轻量级滚动矩稳定性评分，`stationarity_method` 显示 `rolling(n/a-statsmodels)`。
若恰好安装了 `statsmodels`，则自动使用 ADF 检验。

## 示例

```bash
uv run python tool/dataset_characteristics.py \
    --config configs/datasets/etth1.toml \
    --split train --per-channel --period 24 \
    --out work_dirs/ETTh1/characteristics_train.csv
```
