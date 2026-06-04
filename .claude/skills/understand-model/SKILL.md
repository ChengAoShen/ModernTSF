---
name: understand-model
description: Understand a model in ModernTSF by reading its README card as progressive disclosure — paper venue/date/arXiv/abstract first, then source only if needed. Use when the user asks "what is model X", "tell me about X", "了解 X 模型", "how does X work", which paper is X from, or wants to compare/learn what a forecaster does before running it.
---

## What this does

Each model under `src/models/<module>/` ships a **README card** — a compact, self-describing
summary (YAML frontmatter + paper metadata + abstract + description). This skill uses that card
as a **progressive-disclosure** entry point: answer the user from the cheapest layer that
suffices, and only read heavier sources (source code, config, docs) when the question genuinely
needs implementation depth. Do **not** dump `model.py` to explain what a model is — start at the card.

## Step 1 — Resolve the model to its module directory

The user gives a display name (often fuzzy, e.g. "DLinear", "patch tst", "itransformer").

1. Glob `src/models/*/README.md` and grep the frontmatter `model:` field, case-insensitively:
   ```bash
   grep -ril '^model: *"<query>"' src/models/*/README.md   # exact-ish
   grep -ril 'model:.*<query>' src/models/*/README.md       # fuzzy fallback
   ```
   The directory is `src/models/<module>/`.
2. If no match, the authoritative display-name → module map is `MODEL_NAME_MAP` in
   `src/benchmark/registry/models.py` (172 entries), and the catalogue with one-line notes is
   `docs/en/models.md`. Use these to disambiguate.
3. If several candidates match, list them and ask the user which one — don't guess.

## Step 2 — Progressive disclosure: read only as deep as the question needs

Read the card once (`src/models/<module>/README.md`), then answer from the shallowest layer:

| Layer | Read | Answers questions like |
|---|---|---|
| **L1 — Card header** | README frontmatter + `# title` + lead paragraph | "What is X?", "What setting is it for?", "Where's its config?" |
| **L2 — Paper** | README `## Paper` + `## Abstract` | "Which paper / venue / year?", "arXiv link?", "What's the core idea?" |
| **L3 — Interface & params** | `schema.py` + `configs/models/<Name>.toml` | "What hyper-parameters does it take?", "What are the defaults?" |
| **L4 — Implementation** | `model.py` (and `registry.py` for the factory) | "How is it implemented?", "What does forward do?", "How does the adapter reshape inputs?" |

Stop at the first layer that fully answers the question. Most "tell me about X" / "what paper" /
"了解 X" questions are answered entirely at **L1–L2** — one file read. Escalate to L3/L4 only for
implementation or hyper-parameter questions, and say which file you're drawing from.

## Step 3 — Answer

- Lead with the card's own framing (display name, forecasting setting, paper + venue + year, arXiv).
- Quote or paraphrase the abstract for the "core idea". Link the arXiv URL verbatim.
- Reference files as clickable `src/models/<module>/model.py:NN` when you go deeper.
- If the card's paper fields are empty (a classical baseline or an unresolved entry), say so plainly
  rather than inventing a citation — and offer to dig into the source instead.

## Notes

- The card is the contract; trust it first. It carries `paper_title`, `venue`, `year`, `arxiv`,
  the original abstract, and `config` / `registry` pointers in its YAML front matter, followed by
  a lead paragraph and the `## Paper` / `## Abstract` / `## In ModernTSF` sections.
- A `venue` of `N/A (classical baseline)` / `N/A (... in-repo baseline)` means the model has no
  originating paper (a statistical/ML or repo baseline) — that is expected, not a missing card.
- To then **run** the model, hand off to the `run` skill; to **compare** models on a dataset, the
  `rank` / `report` skills; to **add** a new one, the `add-model` skill.
