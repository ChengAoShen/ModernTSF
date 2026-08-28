# ModernTSF 文档

← [项目 README](../../README_zh.md)

人读文档有意保持精简。命令参数和默认值请直接查看 `uv run tsf --help`
以及各子命令的 `--help`；模型、组件和数据集细节由各自卡片承载。

- [workflows.md](workflows.md)：模型接口、模型引入、Foundation artifact、
  组件、数据、验证和实验。
- [models.md](models.md)：自动生成的平铺目录以及全部模型卡链接。

快速开始：

```bash
UV_TORCH_BACKEND=auto uv sync --python 3.12
uv run tsf model list
uv run tsf dataset list
uv run tsf inspect --config configs/runs/run_single_data.toml
uv run tsf run configs/runs/run_single_data.toml
```
