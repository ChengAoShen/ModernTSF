# 模型与组件

ModernTSF 提供两个互相正交的平铺目录。它们是代码职责边界，不是架构分类：

- **模型与方法**是 `src/models/<slug>/` 下可运行的命名条目。每个条目拥有
  `ModelSpec`、参数 schema、preset、实现包装、模型卡、来源证据状态和张量契约。
- **组件**位于 `src/components/`，用于承载多个模型语义和形状完全一致的数学模块。

无需导入具体实现即可查询全部目录：

```bash
uv run tsf model list --details
uv run tsf model show DLinear
uv run tsf component list
uv run tsf component match "patch transformer normalization"
uv run tsf component show quantile_head
```

`tsf model show` 会展示模型卡中的方法简介。组件查询会展示语义契约、公开符号和
使用它的模型包。

`component match` 只根据契约词检索候选，不代表两个实现可以在数学上互换。
复用前必须通过 `component show` 检查候选，并验证其完整语义。

应通过公开 CLI 解析可运行条目，确保构造逻辑、参数校验、证据和 capability
始终保持一致。
