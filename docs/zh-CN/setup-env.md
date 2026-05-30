# 环境配置（硬件感知）

ModernTSF 不再把 **PyTorch 构建版本**（CPU 或某个 CUDA `cuXXX`）写死，而是在
安装时选择。`pyproject.toml` 只声明裸版本 `torch==2.6.0` /
`torchvision==0.21.0` / `torchaudio==2.6.0`，**不带** `+cuXXX` 本地版本号、
**也不带**显式 `[tool.uv.index]`——因此后端由 uv 通过 `UV_TORCH_BACKEND`
环境变量选择。

---

## 工作原理

1. **检测** GPU 以及驱动支持的最高 CUDA 版本。
2. **翻译**为后端标签（`cpu`、`cu118`、`cu121`、`cu124`、`cu126`、`cu128`）。
3. **安装**——`uv sync` 读取 `UV_TORCH_BACKEND`，自动从
   `https://download.pytorch.org/whl/<backend>` 解析 torch。

Agent 不手动下载 wheel：检测硬件 → 设置一个环境变量 → 让 `uv sync` 拉取正确构建。

---

## 步骤 1 — 检测硬件

```bash
bash scripts/detect_hardware.sh
```

示例输出：

```
gpu=NVIDIA GeForce RTX 4090
driver=550.54.15
cuda=12.4
backend=cu124
```

无 GPU / 无 `nvidia-smi` 时报告 `backend=cpu`。用 `--backend` 只打印标签
（便于直接拼进安装命令）。

## 步骤 2 — 按后端安装

推荐——让 uv 自动探测：

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
```

显式指定（可复现，或 `auto` 判断有误时）：

```bash
UV_TORCH_BACKEND=cu124 uv sync --python 3.12
UV_TORCH_BACKEND=cpu   uv sync --python 3.12
```

或直接喂入检测结果：

```bash
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

## 步骤 3 — 验证

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## CUDA ↔ 后端 ↔ torch 对照表

| 驱动 CUDA（`nvidia-smi`） | `UV_TORCH_BACKEND` | torch 版本 |
|---|---|---|
| 无 / 无 `nvidia-smi` | `cpu` | `2.6.0`（任意） |
| 11.8 | `cu118` | `2.6.0` |
| 12.1–12.3 | `cu121` | `2.6.0` |
| 12.4–12.5 | `cu124` | `2.6.0`（项目默认） |
| 12.6–12.7 | `cu126` | `2.6.0` |
| 12.8+ / 13.x（如 Blackwell） | `cu128` | **需 `torch>=2.7`** |

CUDA 在同一主版本内向后兼容，因此较新的驱动可以运行较旧的 `cuXXX` wheel——
选择不超过驱动 CUDA 的最高标签即可。

---

## 不同 CUDA → 不同 PyTorch 版本

项目基线为 `torch==2.6.0`，它提供 `cpu`/`cu118`/`cu121`/`cu124`/`cu126` 的 wheel。
`cu128` 的 wheel 仅在 `torch>=2.7` 才有，因此在需要它的新 GPU 上，先抬升版本：

```bash
uv add "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.7"
UV_TORCH_BACKEND=cu128 uv sync --python 3.12
```

---

## 可复现性与锁文件

- `uv.lock` 记录解析到的 wheel。切换 `UV_TORCH_BACKEND` 后运行 `uv sync`
  会**为该构建刷新锁文件**——这是预期行为。
- 若需冻结、可复现的环境，请提交在**目标硬件上**生成的 `uv.lock`
  （锁文件天然与后端相关）。
- 不要在 `pyproject.toml` 中重新引入 `+cuXXX` 钉版或显式 `[tool.uv.index]`，
  两者都会覆盖 `UV_TORCH_BACKEND`。

---

## 故障排查

| 现象 | 处理 |
|---|---|
| GPU 机器上 `torch.cuda.is_available()` 为 `False` | 用正确的 `UV_TORCH_BACKEND`（非 `cpu`）重跑 `uv sync`；确认 `nvidia-smi` 可用。 |
| Wheel 报错："no wheel for cu128 / torch 2.6.0" | cu128 需 `torch>=2.7`——抬升版本（见上）。 |
| GPU 机器上 `auto` 解析成了 CPU | `nvidia-smi` 不在 `PATH`；显式指定后端。 |
| sync 一直拉 `+cu124` | `pyproject.toml` 里仍残留 `+cuXXX` 钉版或 `[tool.uv.index]`——删除它。 |

Agent 侧的快速路径见 `setup-env` skill，`detect_hardware.sh` 见
[scripts.md](scripts.md)。
