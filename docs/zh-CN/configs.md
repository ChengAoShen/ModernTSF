# 配置读取与使用

ModernTSF 以 TOML 为中心。配置通过 `extends` 组合，通过 Pydantic 进行校验，并可通过 `[sweep]` 展开为多组实验。

## 配置组织

- `configs/base.toml`：全局默认值。
- `configs/datasets/*.toml`：数据集配置。
- `configs/models/*.toml`：模型配置。
- `configs/runs/*.toml`：入口配置，通过 `extends` 组合。

示例入口（`configs/runs/run_single_data.toml`）：

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[evaluation]
enable_profile = true

[sweep.task]
pred_len = [96, 192, 336, 720]
```

## `extends` 合并规则

- 支持单个字符串或列表。
- 路径相对当前配置文件。
- 字典深度合并；标量和列表被后续值覆盖。
- 合并结果由 `RootConfig` 校验。

## sweep 展开

`[sweep]` 会生成笛卡尔积：

```toml
[sweep]
experiment.random_seed = [0, 1, 2]

[sweep.task]
pred_len = [96, 192, 336, 720]
```

每个组合会被解析为一次独立实验。

## sweep.extend（多配置轴）

可以在 `sweep.extend` 中直接引用多个 TOML 文件，生成多组配置组合：

```toml
extends = ["../../base.toml", "../../models/DLinear.toml"]

[sweep.extend]
datasets = [
  "../../datasets/electricity.toml",
  "../../datasets/etth1.toml",
]
models = [
  "../../models/DLinear.toml",
  "../../models/Linear.toml",
]

[sweep.task]
pred_len = [96, 192]
```

- `sweep.extend.<axis>` 轴名可自定义。
- 路径相对当前 run 配置文件解析。
- 所有轴按笛卡尔积组合。
- 合并优先级：`extends` < `sweep.extend` 配置 < 当前 run 配置。
- 结果输出会在 CSV 中记录 `sweep.extend.<axis>` 字段。

## 多重 sweep 策略

`sweep.extend` 会先展开，然后再展开剩余的 `[sweep]` 参数。最终运行数量为：

```text
总运行数 = product(len(values) for each sweep.extend axis)
         * product(len(values) for each sweep key)
```

示例（`configs/runs/multi_sweep.toml`）：

- `sweep.extend` 中 2 个数据集 × 2 个模型
- `sweep` 中 2 个随机种子 × 2 个 pred_len
- 总运行数 = 2 × 2 × 2 × 2 = 16

## 配置预览

可以用辅助脚本查看配置展开结果：

```bash
uv run python tool/inspect_config.py --config configs/runs/multi_sweep.toml
```

脚本会输出总运行数、覆盖的数据集/模型，以及各个 sweep 的取值范围。

## 运行命令

```bash
uv run modern-tsf --config configs/runs/run_single_data.toml
```

输出目录由 `experiment.work_dir` 决定。

每次运行在训练前会输出一行摘要，包含模型、数据集、关键任务参数，以及当前 sweep 的取值。

## 数据集单独配置

可视化脚本可以直接读取只有 `[dataset]` 的配置文件，此时会从 `configs/base.toml` 读取默认的 `task` 参数。
