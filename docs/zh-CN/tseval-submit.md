# 向 TSEval 榜单提交结果

ModernTSF 是实验运行的地方；[**TSEval**](https://tseval.diaugeia.ai) 是把实验公
开展示出来的地方。一次提交不是你往表格里填的一个数字——而是三样东西打成的一个
bundle：你的结果、产生它的 agent **轨迹**，以及一份人类可读的报告。榜单上的每一
行都是一次已提交的 submission，任何人都能打开、审计、复现；榜单是证据的函数，每次
提交都会重建。

唯一的真理源是 GitHub 上的榜单仓库：
[github.com/Diaugeia/TSEval](https://github.com/Diaugeia/TSEval)。你通过向它开一
个 pull request 来贡献一次提交——无需特殊权限，无需手动改表格。本页是生产侧的操作
流程。

## 1. 跑实验（可选：开启轨迹捕获）

建议捕获轨迹：它是审查者审阅的依据。捕获发生在 `tsf` 的 CLI 边界，因此**与具体
agent 无关**（Claude Code / Codex / OpenCode / 人工 都一样）。

```bash
# 开启捕获 session（可选但推荐）
uv run python tool/tsf.py trace start --label "patchtst-etth1-sweep"

# 照常跑实验——每条 tsf 命令都会被记录
uv run python tool/tsf.py run configs/runs/<your_config>.toml

# 结束 session
uv run python tool/tsf.py trace end          # 或: tsf trace status
```

每个 run 会在 `work_dirs/<dataset>/<model>/` 下写入：

- `records/<run_id>.json`——经 schema 校验的 `RunRecord`（自描述：metrics +
  profile + env + git SHA）。这是 `tsf submit` 读取的对象。
- `performance.csv` / `profile.csv`——常规 CSV 输出。

## 2. 本地打包 bundle

```bash
# 本地组装 submission bundle——不上传、不 push
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest
```

`--latest` 取最新一次 run；用 `--run-id <id>` 指定某次。打包结果写到
`work_dirs/_submissions/<submission_id>/`，恰好包含三个文件：

| 文件 | 内容 |
|---|---|
| `submission.json` | `SubmissionReport`（结果 + 数据集规格 + 带 sha256 的文件清单） |
| `trajectory.jsonl` | 捕获到的实验过程（若未捕获则为标记 `synthetic` 的占位） |
| `report.md` | 人类可读摘要（metrics、profile、运行环境） |

submission bundle 里**没有任何权重引用**。一行靠它的结果和过程赢得位置，而不是靠
checkpoint——上榜从不要求上传 `.pth`。如果你*想要*比特级复现，可以选择把训练好的
权重归档到公开的
[TSEval-Weights](https://huggingface.co/datasets/Diaugeia/TSEval-Weights) 数据集，
但这是一份邀请，绝不是门槛。

## 3. 通过 GitHub pull request 贡献

bundle 通过加入榜单仓库的本地 clone 并开 PR 来贡献——没有 `--push`，也不会上传任
何东西到 Hugging Face。

```bash
# clone github.com/Diaugeia/TSEval，然后把你的 bundle 目录放到位：
#   submissions/<track>/<dataset>/<model>/<run_id>/
#     ├── submission.json
#     ├── trajectory.jsonl
#     └── report.md
# 提交并开一个 pull request。
```

CI 会用 TSF-Core JSON Schema 校验你的 bundle，跨 seed 聚合（mean + `n_runs` + 每
个 metric 的 std），如果通过——就把你的行连同完整证据一起重新部署到榜单上。

## 4. 审查与合并

审查者打开 `report.md`、浏览 `trajectory.jsonl`，然后合并 PR。v1 的审查是**人工**
的——不跑自动 agent 核验——轨迹作为证据存档并人工过目。

## 备注

- 没开轨迹 session？`tsf submit` 仍可用，会写一份标记 `synthetic: true` 的最小轨
  迹。但建议捕获真实轨迹。
- 契约（合法提交长什么样）= `tsf_core` 包通过 `tsf schema-export` 导出的 JSON
  Schema——与 TSEval 共享的唯一真理源，零 Python 耦合（消费侧无 torch、不 import
  ModernTSF）。
