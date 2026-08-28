"""Route-neutral verification primitives and subject fingerprinting."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CheckEvidence(_StrictModel):
    """Result and trace for one required executable check."""

    passed: bool
    evidence: list[str] = Field(min_length=1)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("evidence")
    @classmethod
    def _nonempty_evidence(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence entries must not be empty")
        return values


class ExecutionEnvironment(_StrictModel):
    """Minimum reproducibility facts attached to executable evidence."""

    python: str
    framework: str
    dependencies: dict[str, str] = Field(min_length=1)
    platform: str
    device: str
    dtype: str
    deterministic: dict[str, bool | int | str] = Field(min_length=1)

    @model_validator(mode="after")
    def _all_facts_present(self) -> "ExecutionEnvironment":
        values = (self.python, self.framework, self.platform, self.device, self.dtype)
        if any(not value.strip() for value in values):
            raise ValueError("execution environment facts must not be empty")
        if any(not key.strip() or not value.strip() for key, value in self.dependencies.items()):
            raise ValueError("dependency names and versions must not be empty")
        if any(not key.strip() for key in self.deterministic):
            raise ValueError("deterministic setting names must not be empty")
        return self


def _subject_paths(root: Path, fields: dict[str, object]) -> list[Path]:
    paths: set[Path] = set()
    package = root / "src" / "models" / str(fields["package"])
    if package.is_dir():
        paths.update(
            path
            for path in package.rglob("*")
            if path.is_file() and (path.suffix == ".py" or path.name == "README.md")
        )
    config = fields.get("config_path")
    if config:
        paths.add(root / str(config))

    from components.audit import component_dependency_closure
    from components.catalog import COMPONENT_CATALOG

    component_names = component_dependency_closure(
        {str(name) for name in fields.get("components", ())}
    )
    for name in component_names:
        module = COMPONENT_CATALOG.get(name).module
        paths.add(root / "src" / Path(*module.split(".")).with_suffix(".py"))
    return sorted(paths)


def verification_subject_sha256(root: Path, fields: dict[str, object]) -> str:
    """Fingerprint model code, card, config, and declared shared dependencies."""
    root = root.resolve()
    digest = hashlib.sha256()
    for path in _subject_paths(root, fields):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"verification subject escapes repository: {path}") from exc
        if not resolved.is_file():
            raise ValueError(f"verification subject file is missing: {relative}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def evidence_file_sha256(path: Path) -> str:
    """Hash a supporting verification artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
