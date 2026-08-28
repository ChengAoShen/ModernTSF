# 文档

← [返回 ModernTSF](../../README_zh.md)

这些页面面向使用或扩展 ModernTSF 的读者。示例统一使用公开的 `tsf` 命令；
从最符合当前目标的一行开始即可。

## 安装与配置

| 目标 | 指南 |
| --- | --- |
| 安装匹配 CPU 或 CUDA 的环境 | [setup-env.md](setup-env.md) |
| 查询受支持的命令 | [scripts.md](scripts.md) |
| 理解配置继承和 sweep | [configs.md](configs.md) |
| 查询全部配置字段 | [params.md](params.md) |

## 选择模型与数据设定

| 目标 | 指南 |
| --- | --- |
| 浏览全部模型与方法 | [models.md](models.md) |
| 理解模型、组件和适配器 | [modules.md](modules.md) |
| 选择时序、时空或协变量数据设定 | [task-modes.md](task-modes.md) |
| 配置点、分位数或分布输出 | [probabilistic-forecasting.md](probabilistic-forecasting.md) |

## 添加或准备数据和模型

| 目标 | 指南 |
| --- | --- |
| 添加模型或方法 | [add-model.md](add-model.md) |
| 添加数据集 | [add-dataset.md](add-dataset.md) |
| 转换交通图数据 | [datasets-traffic.md](datasets-traffic.md) |
| 预切片 CSV 数据 | [pre-process.md](pre-process.md) |

## 运行与检查实验

| 目标 | 指南 |
| --- | --- |
| 运行实验、sweep 和 case 图 | [experiments.md](experiments.md) |
| 执行前预览解析后的 runs | [inspect-config.md](inspect-config.md) |
| 检查数据集特征 | [dataset-characteristics.md](dataset-characteristics.md) |
| 绘制数据集样本 | [visualize-data.md](visualize-data.md) |
| 运行 GIFT-EVAL | [gift-eval.md](gift-eval.md) |

## 分析与共享结果

| 目标 | 指南 |
| --- | --- |
| 聚合结果文件 | [aggregate-results.md](aggregate-results.md) |
| 对可比较的 runs 排名 | [rank-models.md](rank-models.md) |
| 绘制精度与成本权衡图 | [plot-bubble.md](plot-bubble.md) |
| 打包 TSEval 提交 | [tseval-submit.md](tseval-submit.md) |
