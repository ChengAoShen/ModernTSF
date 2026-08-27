# 公共命令行

ModernTSF 只有一个公开入口：

```bash
uv run tsf --help
```

以下命令构成受支持的用户接口。

## 模型与仓库

```bash
uv run tsf model add --name MyModel --params "enc_in:int,hidden:int=128"
uv run tsf model list
uv run tsf model show DLinear
uv run tsf model audit DLinear
uv run tsf component match "patch transformer normalization"
uv run tsf repo audit
uv run tsf repo doctor --forward
uv run tsf repo doctor --backward
```

`component match` 返回仍需语义审核的候选。`repo audit` 检查仓库一致性；`repo doctor` 还会构造全部模型。`--forward` 会执行登记的张量契约与回归随机种子，`--backward` 还会检查训练梯度是否存在且为有限值，并隐含启用 `--forward`。

## 数据

```bash
uv run tsf dataset add --help
uv run tsf dataset prepare --help
uv run tsf dataset inspect --help
uv run tsf dataset plot --help
uv run tsf dataset convert-traffic --help
uv run tsf dataset gift-download --help
```

## 实验

```bash
uv run tsf inspect --config configs/runs/run_single_data.toml
uv run tsf run configs/runs/run_single_data.toml
uv run tsf run configs/runs/a.toml configs/runs/b.toml --jobs 2 --gpus 0,1
uv run tsf smoke --model DLinear
uv run tsf smoke --all --jobs 8
```

## 结果与提交

```bash
uv run tsf result aggregate --help
uv run tsf result rank --help
uv run tsf result plot --help
uv run tsf result predictions --help
uv run tsf result report --help
uv run tsf trace start --label experiment
uv run tsf submit --dataset ETTh1 --model DLinear --latest
uv run tsf schema-export --check
uv run tsf leaderboard-build --source work_dirs/_submissions
```

使用 `uv run tsf <group> <operation> --help` 查看具体参数。

## 硬件探测

`scripts/detect_hardware.sh` 是只读辅助脚本，用于选择 `UV_TORCH_BACKEND`：

```bash
bash scripts/detect_hardware.sh
UV_TORCH_BACKEND=auto uv sync --python 3.12
```
