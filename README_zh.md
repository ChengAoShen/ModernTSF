<div align="center">

# 🚀 ModernTSF

**现代时间序列预测框架**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Time Series Forecasting](https://img.shields.io/badge/任务-时序预测-blue.svg)](docs/zh-CN/models.md)
[![Models: 100+](https://img.shields.io/badge/模型-100+-orange.svg)](docs/zh-CN/models.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**面向时间序列预测的 AI Infrastructure** —— 而不只是又一个工具包。
一个统一、可复现的底座，让人和 Agent 都把时间花在*创新 idea* 上，
而不是它周围的各种胶水工作。

🗣️ **Clone 本仓库，打开 [Claude Code](https://claude.com/claude-code) 或 [Codex](https://developers.openai.com/codex)，说出你的想法——这就是全部工作流。**

[**English**](README.md) | [**中文**](README_zh.md)

</div>

> 🧪 **最新功能会先在 [`dev`](https://github.com/Diaugeia/ModernTSF/tree/dev) 分支更新。** `main` 是稳定的带版本号发布线；如需尝鲜（测试版）功能，可关注或直接从 `dev` 安装。

---

## 🧭 ModernTSF 是什么

开车不必从造车开始，烤面包不必从磨面粉开始，喝咖啡也不必从种豆子开始——
直接用现成的就好。AI 研究同样需要这样一层基础设施：如今的 Agent 已经能写代码、
跑实验，但无论对人还是 Agent，大部分精力仍然耗在复现已有工作、验证 baseline、
调试环境和编写胶水代码上。ModernTSF 就是时间序列预测领域所缺失的这一层基础设施：
你只需带来 idea，周边的一切由底座兜底。

---

## ✨ 亮点

- 🧠 **100+ 模型、60+ 数据集** —— 从线性基线、Transformer 到图模型与基础模型；经典基准、任意自定义 CSV、交通图、GIFT-EVAL
- 🤖 **Agent 优先** —— 项目指令与 19 个 Agent Skills 由 Claude Code 和 Codex 共用，每个工作流都只需一句话
- 🎛️ **三种数据设定** —— `time_series`、`spatiotemporal`、`covariate`，可按 run 切换
- 🔁 **可复现、可审计** —— 可版本化的 TOML 配置、固定随机种子、带性能分析的输出、可回放的 Agent 轨迹，让结果真正可比
- 🛠️ **统一入口** —— `tsf` 一个命令完成脚手架、smoke 测试、扫描、聚合、排名、绘图与报告

---

## 🏁 如何使用

```bash
git clone https://github.com/Diaugeia/ModernTSF.git
cd ModernTSF
claude        # 或：codex
```

然后用自然语言说出你想做的事即可：

```text
> 帮我按这台机器的 GPU 配好环境。
> 在 ETTh1 上跑 DLinear、PatchTST 和 iTransformer，给我一份排行榜。
> 这是我的小时级销售 CSV——把它接入为数据集，并找出最适合它的模型。
> 我有一个 idea：<描述它>。帮我脚手架一个新模型、实现它，并和强基线对比。
```

Agent 会自动读取项目指令与 Agent Skills——环境配置、脚手架、冒烟测试、
扫描实验、聚合、排名、报告，全部包含在内。

---

## 📖 文档

完整参考都在文档索引里——参数、配置、任务模式、添加模型与数据集、工具，
以及 GIFT-EVAL 基准：

🇬🇧 [English docs](docs/en/README.md) · 🇨🇳 [中文文档](docs/zh-CN/README.md)

> 但我想你不会需要这一切的——让 Agent 来看吧。

---

## 📜 许可证

ModernTSF 以 [MIT 许可证](LICENSE) 发布 — 开放优先，可自由使用、修改与二次开发。

版权所有 © 2026 **Diaugeia.AI**。

仓库内置的第三方模型实现仍遵循其各自的上游许可证；归属信息见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Diaugeia/ModernTSF&type=Date)](https://star-history.com/#Diaugeia/ModernTSF&Date)
