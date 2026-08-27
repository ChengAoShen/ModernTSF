# 添加模型或方法

模型与方法都平铺在 `src/models/<module>/` 下，不建立架构分类目录。目录通过每个公开名称对应的单一 `spec.py` 发现模型；preset 只负责运行配置，不承担注册。

## 生成骨架

```bash
uv run tsf model add --name MyModel \
  --params "enc_in:int,hidden:int=128,dropout:float=0.1"
```

只有模型确实消费数据集邻接矩阵时才使用 `--graph`。命令会生成：

```text
src/models/my_model/
  model.py
  spec.py
  README.md
configs/models/MyModel.toml
configs/runs/smoke_my_model.toml
```

同时会加入惰性的 `MODEL_CATALOG` 引用。

## ModelSpec

`spec.py` 是以下信息的唯一事实源：

- Pydantic `ModelParameterConfig` 与构造工厂；
- 公开名称、配置、模型卡与 smoke case；
- capability 与输出类型；
- 共享 adapter 和 component 依赖；
- 论文、上游 revision、许可证、偏差与证据状态；
- 可执行契约使用的最小任务尺寸与回归随机种子。

工厂接收解析后的根配置以及校验过的 `model.params`。公开前向输入为 `(x_enc, x_mark_enc, x_dec, x_mark_dec)`；点预测返回 `(B, pred_len, C 或 N)`，分位数和分布输出通过 capability 声明额外输出轴。

图与日历输入适配应复用 `components.marks`；与论文无关的通用模块放入 `src/components/`。广义近似后端放入 `src/adapters/`，并必须标记为 `evidence="adaptation"`，不能描述成论文复现。

## 证据

在与论文和权威上游 revision 完成逐项核对前，新条目保持 `unverified`。所有重要差异写入 `deviations`；输出形状正确只是必要条件，不构成复现证据。

## 验证

```bash
uv run tsf model show MyModel
uv run tsf smoke --model MyModel
uv run tsf repo doctor --forward
```

三项结果都必须与 preset 和模型卡一致，才能发布。
