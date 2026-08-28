"""Validate human documentation projections, mirrors, links, and boundaries.

Catches the exact kind of drift that's easy to introduce by hand and easy to
miss in review: stale generated model pages, unindexed guides, language-mirror
drift, and Agent-only implementation details leaking into user documentation.
Pure text/file checks keep this safe and fast enough for every PR.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from benchmark.catalog_metadata import model_records

ROOT = Path(__file__).resolve().parents[3]
DOC_DIRS = [ROOT / "docs" / "en", ROOT / "docs" / "zh-CN"]
OBSOLETE_DOC_TOKENS = (
    "tool/",
    "modern-tsf",
    "MODEL_NAME_MAP",
    "MODEL_REGISTRY",
    "registry.py",
    "schema.py",
    "src/models/module",
    "src/models/_external",
)
AGENT_ONLY_TOKENS = (
    ".agents/",
    ".claude/skills",
    "AGENTS.md",
    "CLAUDE.md",
    "SKILL.md",
    "benchmark.commands",
)


def render_models_doc(language: str) -> str:
    records = model_records(ROOT)
    if language == "en":
        intro = (
            "# Models and methods\n\n"
            f"ModernTSF exposes {len(records)} model and method entries through one flat "
            "public catalog. There are no user-facing architecture families. Presets "
            "configure runs and do not create additional entries.\n\n"
            "Implementation origin is declared as `upstream` or `rewrite`; executable "
            "audit and parity gates determine whether that declaration is release-ready.\n\n"
            "| Name | Preset | Implementation | Capabilities | Model card |\n"
            "|---|---|---|---|---|\n"
        )
    else:
        intro = (
            "# 模型与方法\n\n"
            f"ModernTSF 通过单一、平铺的公开目录提供 {len(records)} 个模型和方法条目。"
            "用户侧不设置架构族分类；preset 只负责配置运行，不会创建额外条目。\n\n"
            "实现来源只声明为 `upstream` 或 `rewrite`；可执行审计与数值 parity 门禁负责判断"
            "该声明是否达到发布要求。\n\n"
            "| 名称 | Preset | 实现来源 | Capabilities | 模型卡 |\n"
            "|---|---|---|---|---|\n"
        )
    rows = []
    for record in records:
        name = str(record["name"])
        config = str(record["config_path"])
        package = str(record["package"])
        implementation = str(record["implementation"])
        capabilities = ", ".join(sorted(record.get("capabilities", ()))) or "—"
        rows.append(
            f"| `{name}` | [`{config}`](../../{config}) | `{implementation}` | "
            f"{capabilities} | [README](../../src/models/{package}/README.md) |"
        )
    return intro + "\n".join(rows) + "\n"


def _generated_model_doc_problems() -> list[str]:
    problems: list[str] = []
    for language in ("en", "zh-CN"):
        path = ROOT / "docs" / language / "models.md"
        if path.read_text(encoding="utf-8") != render_models_doc(language):
            problems.append(f"{path.relative_to(ROOT)} is stale; regenerate from ModelSpec")
    return problems


def write_generated_model_docs() -> None:
    """Refresh the two human-readable model catalog projections."""
    for language in ("en", "zh-CN"):
        path = ROOT / "docs" / language / "models.md"
        path.write_text(render_models_doc(language), encoding="utf-8")


def _docs_index_problems() -> list[str]:
    problems = []
    for doc_dir in DOC_DIRS:
        readme = doc_dir / "README.md"
        readme_text = readme.read_text()
        linked = set(re.findall(r"\[([a-zA-Z0-9_.-]+\.md)\]", readme_text))
        for page in doc_dir.glob("*.md"):
            if page.name == "README.md":
                continue
            if page.name not in linked:
                problems.append(
                    f"{page.relative_to(ROOT)} has no link in {readme.relative_to(ROOT)}"
                )
    return problems


def _docs_mirror_problems() -> list[str]:
    en_dir, zh_dir = DOC_DIRS
    en_pages = {p.name for p in en_dir.glob("*.md") if p.name != "README.md"}
    zh_pages = {p.name for p in zh_dir.glob("*.md") if p.name != "README.md"}

    problems = []
    for name in sorted(en_pages - zh_pages):
        problems.append(f"docs/en/{name} has no docs/zh-CN/{name} mirror")
    for name in sorted(zh_pages - en_pages):
        problems.append(f"docs/zh-CN/{name} has no docs/en/{name} mirror")
    return problems


def _obsolete_reference_problems() -> list[str]:
    problems: list[str] = []
    paths = [ROOT / "README.md", ROOT / "README_zh.md"]
    for doc_dir in DOC_DIRS:
        paths.extend(doc_dir.glob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in OBSOLETE_DOC_TOKENS:
            if token in text:
                problems.append(f"{path.relative_to(ROOT)} references obsolete {token!r}")
    return problems


def _audience_boundary_problems() -> list[str]:
    problems: list[str] = []
    human_paths = [ROOT / "README.md", ROOT / "README_zh.md", ROOT / "CONTRIBUTING.md"]
    for doc_dir in DOC_DIRS:
        human_paths.extend(doc_dir.glob("*.md"))
    for path in human_paths:
        text = path.read_text(encoding="utf-8")
        for token in AGENT_ONLY_TOKENS:
            if token in text:
                problems.append(
                    f"{path.relative_to(ROOT)} exposes Agent-only detail {token!r}"
                )

    for path in (ROOT / ".agents").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for token in ("docs/en/", "docs/zh-CN/", "CONTRIBUTING.md"):
            if token in text:
                problems.append(
                    f"{path.relative_to(ROOT)} depends on human documentation {token!r}"
                )
    return problems


def _relative_link_problems() -> list[str]:
    problems: list[str] = []
    paths = [
        ROOT / "README.md",
        ROOT / "README_zh.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CHANGELOG.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]
    for doc_dir in DOC_DIRS:
        paths.extend(doc_dir.glob("*.md"))
    paths.extend((ROOT / ".agents").rglob("*.md"))
    paths.extend((ROOT / "src" / "models").glob("*/README.md"))

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\]\(([^)]+)\)", text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("#", "/", "mailto:")):
                continue
            resolved = path.parent / target
            if not resolved.exists():
                problems.append(
                    f"{path.relative_to(ROOT)} has broken relative link {raw_target!r}"
                )
    return problems


def check() -> list[str]:
    return (
        _generated_model_doc_problems()
        + _docs_index_problems()
        + _docs_mirror_problems()
        + _obsolete_reference_problems()
        + _audience_boundary_problems()
        + _relative_link_problems()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the bilingual model catalog before checking",
    )
    args = parser.parse_args()
    if args.write:
        write_generated_model_docs()
    problems = check()
    if not problems:
        print("OK: generated model docs and mirrored indexes are consistent.")
        return 0
    print(f"Found {len(problems)} doc inconsistency(ies):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
