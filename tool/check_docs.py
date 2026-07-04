"""Static consistency check for docs and model counts.

Catches the exact kind of drift that's easy to introduce by hand and easy to
miss in review: a stale model count left in CLAUDE.md/docs/en/models.md after
adding/removing a model, or a new docs/*.md page that never got linked from
the README index. Pure text/file checks, no imports beyond the standard
library, so it's safe and fast enough to run on every PR.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_DIR = ROOT / "configs" / "models"
DOC_DIRS = [ROOT / "docs" / "en", ROOT / "docs" / "zh-CN"]


def _model_count_problems() -> list[str]:
    actual = len(list(MODEL_CONFIG_DIR.glob("*.toml")))
    problems = []

    claude_md = (ROOT / "CLAUDE.md").read_text()
    m = re.search(r"^## Available Models \((\d+)\)", claude_md, re.MULTILINE)
    if m and int(m.group(1)) != actual:
        problems.append(
            f"CLAUDE.md heading says {m.group(1)} models, but "
            f"configs/models/*.toml has {actual}"
        )

    table = re.search(
        r"^## Available Models.*?\n(\|.*\n)+", claude_md, re.MULTILINE
    )
    if table:
        rows = re.findall(r"^\|\s*[^|]+\|\s*\*?\*?(\d+)\*?\*?\s*\|$", table.group(0), re.MULTILINE)
        if rows:
            category_sum = sum(int(r) for r in rows[:-1])
            table_total = int(rows[-1])
            if table_total != actual:
                problems.append(
                    f"CLAUDE.md's Available Models table Total row says "
                    f"{table_total}, but configs/models/*.toml has {actual}"
                )
            if category_sum != table_total:
                problems.append(
                    f"CLAUDE.md's Available Models category counts sum to "
                    f"{category_sum}, but the table's Total row says {table_total}"
                )

    models_md = ROOT / "docs" / "en" / "models.md"
    m = re.search(r"includes (\d+) models", models_md.read_text())
    if m and int(m.group(1)) != actual:
        problems.append(
            f"docs/en/models.md says 'includes {m.group(1)} models', but "
            f"configs/models/*.toml has {actual}"
        )

    return problems


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


def check() -> list[str]:
    return _model_count_problems() + _docs_index_problems() + _docs_mirror_problems()


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    problems = check()
    if not problems:
        print("OK: model counts and docs index/mirror are consistent.")
        return 0
    print(f"Found {len(problems)} doc inconsistency(ies):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
