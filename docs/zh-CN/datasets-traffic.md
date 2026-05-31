# 交通节点 + 邻接数据集（METR-LA、PEMS-BAY、PEMS0x）

这些是时空交通图：一个 `(T, N)` 数值矩阵（`N` 个传感器上的速度或流量）加上一个
`(N, N)` 邻接矩阵。ModernTSF 通过现有的 **`cauair_st`** 节点加载器读取它们——
*没有*专门的交通加载器。先用 `tool/convert_traffic.py` 把原始数组转换成节点
*bundle*，再让随附的某个配置指向输出目录。

配置（均使用 `name = "cauair_st"`、`input_dim = 3`、`scale = false`）：

| 配置 | `root_path` |
|---|---|
| `configs/datasets/metr_la.toml`  | `./dataset/metr_la` |
| `configs/datasets/pems_bay.toml` | `./dataset/pems_bay` |
| `configs/datasets/pems03.toml`   | `./dataset/pems03` |
| `configs/datasets/pems04.toml`   | `./dataset/pems04` |
| `configs/datasets/pems07.toml`   | `./dataset/pems07` |
| `configs/datasets/pems08.toml`   | `./dataset/pems08` |

`input_dim = 3` 保留数值通道以及 `--add-time` 追加的两个日历协变量
（`time_in_day`、`day_in_week`）。`scale = false` 保持这些协变量为原始值——
`time_in_day` / `day_in_week` 已在 `[0, 1)` 区间内，*不应*再做 z-score。
（若想对数值通道做 z-score，请在上游处理，或另写一份翻转 `scale` 的配置。）

## 原始数据获取

数据按需下载；ModernTSF 不附带任何副本。常见来源：

- **METR-LA / PEMS-BAY** —— DCRNN 发布
  （<https://github.com/liyaguang/DCRNN>）提供 `metr-la.h5` / `pems-bay.h5`
  以及 `adj_mx.pkl` / `adj_mx_bay.pkl`。该仓库链接的 Google Drive 镜像是常用下载点。
  把 `.h5` 转成 `(T, N)` 矩阵（例如 `pandas.read_hdf(...).values`）并存为 `.npz`。
- **PEMS03 / PEMS04 / PEMS07 / PEMS08** —— ASTGCN / STSGCN 发布
  （<https://github.com/Davidham3/ASTGCN>、<https://github.com/Davidham3/STSGCN>）
  各自提供 `pems0x.npz`（键 `data`，形状 `(T, N, F)`；通道 0 为流量），以及可用于
  构建邻接的距离 CSV，或预构建的 `adj_mx.npy`。

## 转换成 bundle

```bash
# METR-LA：(T, N) 速度矩阵 + .pkl 邻接（3 元组）
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr-la.npz --values-key data \
    --adj    dataset/metr_la/adj_mx.pkl \
    --output-dir dataset/metr_la \
    --add-time --freq-min 5 \
    --seq-len 96 --pred-len 96

# PEMS04：(T, N, F) 流量张量（取通道 0）+ .npy 邻接
uv run python tool/convert_traffic.py \
    --values dataset/pems04/pems04.npz --values-key data \
    --adj    dataset/pems04/adj_mx.npy \
    --output-dir dataset/pems04 \
    --add-time --freq-min 5 \
    --seq-len 96 --pred-len 96
```

转换器会写出 `cauair_st` 加载器所需的 bundle：

```
<output-dir>/his.npz        # data (T, N, C)、mean (C,)、std (C,)
<output-dir>/adj_mx.npy      # (N, N) 邻接（图模型用）
<output-dir>/idx_train.npy   # 窗口中心索引
<output-dir>/idx_val.npy
<output-dir>/idx_test.npy
```

### 邻接格式

`--adj` 接受 `.npy`、`.npz`（配合 `--adj-key`）或 `.pkl`/`.pickle`。METR-LA 和
PEMS-BAY 的 `adj_mx.pkl` 是 3 元组 `(sensor_ids, sensor_id_to_idx, adj_mx)`；
转换器加载该 pickle 并取**最后一个**元素作为 `(N, N)` 矩阵。若 pickle 本身就是
裸 `ndarray`，则直接使用。Python 2 序列化的文件会回退到 `latin1` 解码。

### 划分比例约定

窗口中心按时间**顺序**划分（不打乱），通过 `--splits` 控制，默认
`0.7,0.1,0.2`（训练 / 验证 / 测试），因此测试窗口始终覆盖最新数据。这与
ModernTSF 单文件约定（`split_ratio = [0.7, 0.1, 0.2]`）一致。METR-LA / PEMS-BAY
的 DCRNN 标准划分同样是 `0.7 / 0.1 / 0.2`，故默认即可复现；若需其他约定，请显式
传入 `--splits`。

## 运行

CLI 只接受单个 `--config`，因此通过 `extends` 让 run 配置指向该数据集。例如一个
run 配置（`configs/runs/your_traffic_run.toml`）：

```toml
extends = ["../base.toml", "../datasets/metr_la.toml", "../models/GWNet.toml"]

[task]
mode = "spatiotemporal"
```

然后运行：

```bash
uv run modern-tsf --config configs/runs/your_traffic_run.toml
```
