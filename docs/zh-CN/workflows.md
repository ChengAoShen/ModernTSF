# 工作流与架构

ModernTSF 只有一个平铺模型目录、一个共享组件区、一条数据流水线和一条
verification 路线。模型与 method 是同级条目。

```text
src/models/<slug>/              本地模型代码、spec 和模型卡
src/models/_components/<name>/  可复用组件及组件卡
src/data/                       数据加载器和参数 schema
dataset/                        本地数据文件，不参与打包
catalog/datasets/<preset>/      自动生成的数据集卡
configs/                        可组合的模型、数据集和运行 TOML
verification/                   清单、自动索引和逐模型 evidence
work_dirs/                      实验 checkpoint、指标和运行记录
```

## 模型运行接口

所有模型接收同一个调用：

```python
forecast = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

- `x_enc`：历史观测，通常为 `[batch, seq_len, channels]`；
- `x_mark_enc`：历史时间或节点协变量，也可以为 `None`；
- `x_dec`：decoder 历史前缀与未来占位；
- `x_mark_dec`：已知未来协变量，也可以为 `None`。

模型可以忽略方法不需要的可选输入，但必须接受完整接口。点预测输出为
`[batch, pred_len, targets]`；分位数和分布输出额外增加参数轴。`spec.py`
只保存构造函数、参数 schema、preset、task 能力、共享组件、artifact、可选训练
目标和运行 fixture；描述性事实全部写入 README。

通用 trainer 负责 batch、四输入调用、配置 loss、callback、优化和验证。论文特有
的完整训练目标必须通过 `ModelSpec.training_objective` 显式声明；适配器在一次
计算中返回预测和有限标量 loss。不要把训练目标伪装成 capability，也不要留下
runner 不会调用的 `training_loss()`。验证集和测试指标始终使用配置的观测 loss。

`task.mode` 会在实验启动前验证：

- `time_series`：多变量值和可选日历特征；
- `spatiotemporal`：节点值与历史节点/时间协变量；
- `covariate`：节点值与已知未来协变量。

通过 `tsf model show <Name>` 和 `tsf dataset show <preset>` 查看兼容性。

## 加入模型或方法

引入流程分成 workspace 和 admission 两阶段，placeholder 无法直接进入目录。

1. 使用 `tsf model list --json`、`tsf model search` 对论文去重。
2. 阅读论文和 supplement；有官方代码时记录许可证、固定 revision，并用它
   补足论文未说明的 tensor 顺序、padding、初始化和默认值。不要复制或依赖
   外部模型源码。
3. 将每个关键操作判定为复用现有组件、抽取严格等价的新组件或保留在模型内：

   ```bash
   uv run tsf component match "operation and tensor contract" --json
   uv run tsf component show <candidate>
   ```

4. 创建尚未注册的 workspace：

   ```bash
   uv run tsf model scaffold \
     --name MyModel \
     --paper-title "Paper title" \
     --paper-url https://arxiv.org/abs/0000.00000 \
     --venue Conference --year 2026 \
     --code-url https://github.com/org/repo \
     --revision 0123456789abcdef \
     --license Apache-2.0 \
     --components revin,flatten_forecast_head \
     --params "enc_in:int,d_model:int=128"
   ```

   没有官方代码时同时省略三个 code 参数。只有完成组件检索并确认无匹配时才
   使用 `--components none`；按需指定 `--task-mode spatiotemporal` 或
   `covariate`。
5. 删除所有 scaffold 标记，本地编写实现，保留有价值的论文公式和注释，补全
   模型卡，并在 `verification/models.toml` 声明论文/公式测试。有官方代码时
   必须提供 reference-comparison 测试；无官方代码时该检查为 `not-applicable`。
6. 执行 admission：

   ```bash
   uv run tsf model add --name MyModel
   ```

   命令会临时注册模型，执行统一 verification、单模型审计、严格运行契约、
   组件审计和仓库审计；任何 gate 失败都会回滚注册。

## Foundation Model 与 artifact

Foundation Model 仍是普通的平铺模型条目，预训练规模不是目录分类。只有本地
架构而没有发布权重时，模型卡必须明确说明，不能声称具备对应 checkpoint 的
zero-shot 行为。

权重、tokenizer 和归一化统计通过 `spec.py` 中的 `ModelArtifact` 声明：

```python
from benchmark.registry.models import ModelArtifact

artifacts=(ModelArtifact(
    name="weights",
    url="https://host/repository/resolve/full-commit-or-release/weights.safetensors",
    revision="full-commit-or-release",
    sha256="<64 位小写十六进制>",
    filename="weights.safetensors",
    required=True,
),)
```

构造模型时绝不会隐式下载。用户必须显式检查和获取：

```bash
uv run tsf model artifacts MyFoundationModel
uv run tsf model artifacts MyFoundationModel --fetch weights
```

默认使用用户缓存目录，也可以通过 `MODERNTSF_CACHE` 修改。required artifact
必须存在且 SHA-256 一致，factory 才会运行。加载、离线失败和 checkpoint 能力
声明都需要对应测试和模型卡说明。

## 组件

组件只放在 `src/models/_components/<name>/`。每个组件具备实现、目录契约、
聚焦测试和 README 卡片。只有数学语义和运行语义严格等价才能抽取；同名或相同
tensor rank 不足以证明可复用。必须核对轴、归一化、mask、残差顺序、初始化、
状态、输出、梯度和序列化。

```bash
uv run tsf component list
uv run tsf component match "patch forecast head" --json
uv run tsf component show flatten_forecast_head
uv run tsf component audit
```

论文特有模块保留在模型目录，命名模型之间不能互相导入实现代码。

## 数据

数据严格分为三层：

- `dataset/`：本地下载和转换后的数据，不放代码或卡片；
- `src/data/`：加载器、基础契约和 Pydantic 参数 schema；
- `catalog/datasets/`：每个可运行 preset 的自动生成 README 卡。

```bash
uv run tsf dataset add --name my_data --pattern custom \
  --root-path ./dataset/my_data --data-path my_data.csv --target OT
uv run tsf dataset inspect --config configs/datasets/my_data.toml
uv run tsf dataset show my_data
uv run tsf dataset audit
```

`dataset prepare`、`convert-traffic` 和 `gift-download` 提供显式转换或下载；
写入前先查看各自 `--help`。缩放只能拟合训练集，split 边界必须稳定，图和协变量
loader 必须声明支持的 task mode。

## 实验

run TOML 组合 base、dataset 和 model preset。科研选择全部写入配置，而不是临时
shell 脚本：

```toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[experiment]
description = "DLinear ETTh1 baseline"
random_seed = 42
work_dir = "./work_dirs"

[task]
mode = "time_series"
seq_len = 96
label_len = 0
pred_len = 96
features = "M"

[sweep]
experiment.random_seed = [0, 1, 2]
task.pred_len = [96, 192]
```

运行昂贵实验前先检查完整矩阵：

```bash
uv run tsf inspect --config configs/runs/<run>.toml
uv run tsf run configs/runs/<run>.toml
```

确认资源后才使用 `--jobs` 和 `--gpus`。每次运行在 `work_dirs/` 保存解析后的
配置、seed、环境、checkpoint、原始指标和失败记录。只有数据 split、预处理、
horizon、指标定义和评估策略一致的 cell 才能比较。使用 `tsf result --help`
查看聚合、排名、绘图、预测检查和报告命令。

## Verification 与仓库 gate

模型只使用一套结构：

```text
verification/models.toml
verification/index.json
verification/evidence/<Model>.json
```

evidence 检查论文结构、公式、构造、forward、backward、有限输出、活跃参数梯度、
state-dict round trip、CPU、batch/序列边界、输入契约和 reference comparison；
index 只能自动生成。

```bash
uv run tsf verify model DLinear
uv run tsf verify stale
uv run tsf verify all --jobs 8
uv run tsf verify index
uv run tsf model audit --summary
uv run tsf repo doctor --strict
uv run tsf repo audit
```

论文结果复现与代码 verification 分离：通过 run 配置对齐数据集、split、预处理、
优化器、seed、指标和表格 cell，并如实列出偏差；一次成功 forward 不等于复现论文。
