# 向 TSEval 榜单提交结果

ModernTSF 产出结果，**TSEval**（在 Hugging Face 上）负责收集。一次提交是一份自
包含的 **Submission Report**——你的结果 + 产生它的 agent **轨迹** + 一份人类可读
报告——以 PR 形式上传。本页是生产侧的操作流程。

## 1. 跑实验（可选：开启轨迹捕获）

建议捕获轨迹：它是维护者审查的依据。捕获发生在 `tsf` 的 CLI 边界，因此**与具体
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

## 2. 打包并提交

```bash
# 本地打包（dry build），不上传
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest

# 向 HF Submissions 数据集开 PR
uv run python tool/tsf.py submit --dataset <DATASET> --model <MODEL> --latest --push
```

`--latest` 取最新一次 run；用 `--run-id <id>` 指定某次。打包结果写到
`work_dirs/_submissions/<submission_id>/`，包含：

| 文件 | 内容 |
|---|---|
| `submission.json` | `SubmissionReport`（结果 + 数据集规格 + 带 sha256 的文件清单） |
| `trajectory.jsonl` | 捕获到的实验过程（若未捕获则为标记 `synthetic` 的占位） |
| `report.html` | 人类可读摘要（metrics、profile、运行环境） |

`--push` 会向 `Diaugeia/TSEval-Submissions` 开 PR。需要对 `Diaugeia` 组织有 write
权限的 Hugging Face 登录（`hf auth login`，或设置 `HF_TOKEN`）。

## 3. 审查与合并

维护者打开 `report.html`、浏览 `trajectory.jsonl`，然后合并 PR。v1 的审查是
**人工**的——不跑自动 agent 核验——轨迹作为证据存档并人工过目。

## 备注

- 没开轨迹 session？`tsf submit` 仍可用，会写一份标记 `synthetic: true` 的最小
  轨迹。但建议捕获真实轨迹。
- 契约（合法提交长什么样）= `tsf_core` 包通过 `tsf schema-export` 导出的 JSON
  Schema——与 TSEval 共享的唯一真理源。
