# GIFT-EVAL 基准测试

ModernTSF 原生支持 [GIFT-EVAL](https://huggingface.co/datasets/Salesforce/GiftEval) 基准测试 —— 涵盖 23 个基础数据集的 53 个配置，横跨 10 种时间频率（从秒级到月级）以及能源、交通、天气、金融等多个领域。

---

## 第一步 — 下载或链接数据

### 下载全部 53 个数据集

```bash
uv run python tool/gift_eval_download.py --output-dir /your/path
```

从 HuggingFace Hub 上的 `Salesforce/GiftEval` 下载数据至 `--output-dir`，并在 `./dataset/gift_eval` 创建指向该目录的符号链接。默认下载位置为 `~/.cache/gift_eval`。

### 仅下载指定数据集

```bash
uv run python tool/gift_eval_download.py --output-dir /your/path --datasets electricity/15T ett1/H m4_monthly
```

可传入 53 个数据集名称中的任意子集。使用 `--list` 打印所有合法名称：

```bash
uv run python tool/gift_eval_download.py --list
```

### 链接已有数据目录

若数据已在本地，可跳过下载，仅创建符号链接：

```bash
uv run python tool/gift_eval_download.py --link-only --output-dir /path/to/existing/gift_eval
```

---

## 第二步 — 运行 sweep

```bash
uv run modern-tsf --config configs/runs/gift_eval_sweep.toml
```

`gift_eval_sweep.toml` 继承基础配置与 `DLinear.toml`，并通过 `[sweep.extend]` 依次遍历 `configs/datasets/gift_eval/` 下全部 53 个数据集 TOML。将 `DLinear.toml` 替换为任意模型配置，或添加 `[sweep]` 键以同时 sweep 多个模型。

运行前预览展开结果：

```bash
uv run python tool/inspect_config.py --config configs/runs/gift_eval_sweep.toml
```

---

## 预测长度档位

GIFT-EVAL 为每个数据集定义了三档预测长度。`configs/datasets/gift_eval/` 中每个配置文件默认使用**短期（short）**档，并在注释中标注其余两档：

```toml
# pred_len follows GIFT-EVAL "short" term (1x).
# Other terms:  medium = 480 (10x),  long = 720 (15x)

[task]
pred_len = 48
```

如需运行中期或长期，修改对应数据集 TOML 中的 `pred_len`（或在 run config 的 `[sweep]` 中覆盖）。

---

## 数据集列表

| 数据集 | 频率 |
|---|---|
| `electricity` | 15T, D, H, W |
| `ett1`, `ett2` | 15T, D, H, W |
| `solar` | 10T, D, H, W |
| `LOOP_SEATTLE` | 5T, D, H |
| `M_DENSE` | D, H |
| `SZ_TAXI` | 15T, H |
| `bitbrains_fast_storage` | 5T, H |
| `bitbrains_rnd` | 5T, H |
| `bizitobs_application` | — |
| `bizitobs_l2c` | 5T, H |
| `bizitobs_service` | — |
| `hierarchical_sales` | D, W |
| `kdd_cup_2018_with_missing` | D, H |
| `saugeenday` | D, M, W |
| `us_births` | D, M, W |
| `m4_daily`, `m4_hourly`, `m4_monthly` | — |
| `m4_quarterly`, `m4_weekly`, `m4_yearly` | — |
| `car_parts_with_missing` | — |
| `covid_deaths` | — |
| `hospital` | — |
| `jena_weather` | — |
| `restaurant` | — |
| `temperature_rain_with_missing` | — |

---

## 脚本参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-dir DIR` | `~/.cache/gift_eval` | 下载目标路径 / 符号链接目标 |
| `--datasets NAME...` | 全部 53 个 | 仅下载指定数据集 |
| `--link-only` | 关闭 | 跳过下载，仅创建符号链接 |
| `--list` | 关闭 | 打印全部 53 个数据集名称后退出 |

---

## 注意事项

- 符号链接创建于 `./dataset/gift_eval`。所有数据集 TOML 均使用 `root_path = "./dataset/gift_eval"`，无需额外配置即可运行。
- 若 `./dataset/gift_eval` 已作为普通目录存在（非符号链接），脚本将打印警告并跳过符号链接创建。
- 已下载的数据集通过 `dataset_info.json` 的存在来判断，并自动跳过。
- 需安装 `huggingface_hub`（`uv sync` 已包含）。
