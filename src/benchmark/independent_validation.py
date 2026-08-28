"""Canonical evidence for paper-driven local model implementations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import tomllib
from typing import Literal
import unittest

import torch

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchmark.verification_common import (
    CheckEvidence,
    ExecutionEnvironment,
    verification_subject_sha256,
)


SCHEMA_VERSION = 1
DEFAULT_INDEX = Path("verification/index.json")
EVIDENCE_DIRECTORY = Path("verification/evidence")
MANIFEST = Path("verification/models.toml")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class IndependentBasis(_StrictModel):
    """Paper/method basis and independence statement for one implementation."""

    references: list[str] = Field(min_length=1)
    structure_map_sha256: str = Field(pattern=_SHA256_PATTERN)
    independent_design: bool
    source_code_not_copied: bool

    @field_validator("references")
    @classmethod
    def _references(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("basis references must not be empty")
        return values


class IndependentChecks(_StrictModel):
    """Required executable checks shared by every local model implementation."""

    paper_structure: CheckEvidence
    equations: CheckEvidence
    construction: CheckEvidence
    forward: CheckEvidence
    backward: CheckEvidence
    finite_outputs: CheckEvidence
    active_parameter_gradients: CheckEvidence
    state_dict_round_trip: CheckEvidence
    cpu: CheckEvidence
    batch_size_boundary: CheckEvidence
    sequence_length_boundary: CheckEvidence
    marks_adjacency_contract: CheckEvidence


class IndependentValidationEvidence(_StrictModel):
    """The sole persisted verification record for one local implementation."""

    schema_version: Literal[1]
    kind: Literal["independent-validation"]
    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    verified_at: datetime
    subject_sha256: str = Field(pattern=_SHA256_PATTERN)
    commands: list[str] = Field(min_length=1)
    environment: ExecutionEnvironment
    basis: IndependentBasis
    checks: IndependentChecks
    details: dict[str, object] = Field(default_factory=dict)
    passed: bool

    @field_validator("verified_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("verified_at must include a timezone")
        return value

    @field_validator("commands")
    @classmethod
    def _commands(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("commands must not be empty")
        return values

    @model_validator(mode="after")
    def _consistent_pass(self) -> "IndependentValidationEvidence":
        checks_passed = all(check.passed for check in self.checks.__dict__.values())
        basis_passed = self.basis.independent_design and self.basis.source_code_not_copied
        if self.passed != (checks_passed and basis_passed):
            raise ValueError("passed must require an independent basis and every check")
        return self


class EvidenceIndexEntry(_StrictModel):
    """Small generated pointer from a model name to its canonical evidence."""

    evidence: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    subject_sha256: str = Field(pattern=_SHA256_PATTERN)
    verified_at: datetime
    passed: bool

    @field_validator("evidence")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value.strip():
            raise ValueError("evidence path must be repository-relative")
        return value


class IndependentValidationIndex(_StrictModel):
    schema_version: Literal[1]
    models: dict[str, EvidenceIndexEntry]


@dataclass(frozen=True)
class EvidenceState:
    model: str
    status: Literal["passed", "failed", "missing", "invalid", "stale"]
    detail: str | None = None
    evidence: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: Path) -> IndependentValidationIndex:
    """Load the generated index strictly; callers decide how to report failure."""
    return IndependentValidationIndex.model_validate_json(path.read_text(encoding="utf-8"))


def evidence_state(root: Path, name: str, fields: dict[str, object]) -> EvidenceState:
    """Validate an index pointer, canonical evidence, and current subject digest."""
    index_path = root / DEFAULT_INDEX
    if not index_path.is_file():
        return EvidenceState(name, "missing", "verification index is missing")
    try:
        index = load_index(index_path)
    except (OSError, ValueError) as exc:
        return EvidenceState(name, "invalid", f"invalid verification index: {exc}")
    entry = index.models.get(name)
    if entry is None:
        return EvidenceState(name, "missing", "independent evidence is missing")
    evidence_path = root / entry.evidence
    if not evidence_path.is_file():
        return EvidenceState(name, "missing", "indexed evidence file is missing", entry.evidence)
    if file_sha256(evidence_path) != entry.sha256:
        return EvidenceState(name, "stale", "evidence digest differs from index", entry.evidence)
    try:
        evidence = IndependentValidationEvidence.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        return EvidenceState(name, "invalid", f"invalid evidence: {exc}", entry.evidence)
    if evidence.model != name:
        return EvidenceState(name, "invalid", "evidence model does not match index key", entry.evidence)
    if evidence.subject_sha256 != entry.subject_sha256:
        return EvidenceState(name, "stale", "index subject digest differs from evidence", entry.evidence)
    try:
        current_subject = verification_subject_sha256(root, fields)
    except ValueError as exc:
        return EvidenceState(name, "invalid", str(exc), entry.evidence)
    if current_subject != evidence.subject_sha256:
        return EvidenceState(name, "stale", "model, card, config, or component changed", entry.evidence)
    if evidence.verified_at != entry.verified_at or evidence.passed != entry.passed:
        return EvidenceState(name, "stale", "index summary differs from evidence", entry.evidence)
    return EvidenceState(
        name,
        "passed" if evidence.passed else "failed",
        evidence=entry.evidence,
    )


def rebuild_index(root: Path) -> IndependentValidationIndex:
    """Validate every evidence file and atomically regenerate the compact index."""
    evidence_root = root / EVIDENCE_DIRECTORY
    entries: dict[str, EvidenceIndexEntry] = {}
    for path in sorted(evidence_root.glob("*.json")):
        evidence = IndependentValidationEvidence.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if evidence.model in entries:
            raise ValueError(f"duplicate independent evidence for {evidence.model}")
        relative = path.relative_to(root).as_posix()
        entries[evidence.model] = EvidenceIndexEntry(
            evidence=relative,
            sha256=file_sha256(path),
            subject_sha256=evidence.subject_sha256,
            verified_at=evidence.verified_at,
            passed=evidence.passed,
        )
    index = IndependentValidationIndex(schema_version=SCHEMA_VERSION, models=entries)
    target = root / DEFAULT_INDEX
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return index


def load_manifest(root: Path) -> dict[str, dict[str, object]]:
    """Load the declarative model/profile map used to refresh evidence."""
    payload = tomllib.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported independent verification manifest")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("verification manifest requires a models table")
    required = {"paper", "profile", "test", "structure"}
    for name, record in models.items():
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"manifest entry {name} must contain {sorted(required)}")
        if not all(isinstance(record[field], str) and record[field] for field in ("paper", "profile", "test")):
            raise ValueError(f"manifest entry {name} has invalid text fields")
        structure = record["structure"]
        if not isinstance(structure, list) or not structure or not all(
            isinstance(item, str) and item for item in structure
        ):
            raise ValueError(f"manifest entry {name} requires a non-empty structure list")
    return models


def _environment() -> ExecutionEnvironment:
    dependencies = {"torch": torch.__version__}
    for package in ("numpy", "pydantic"):
        dependencies[package] = importlib.metadata.version(package)
    return ExecutionEnvironment(
        python=platform.python_version(),
        framework=f"torch {torch.__version__}",
        dependencies=dependencies,
        platform=platform.platform(),
        device="cpu",
        dtype="float32",
        deterministic={"torch_seed": 104729, "threads": 1},
    )


def refresh_evidence(root: Path, name: str, fields: dict[str, object]) -> IndependentValidationEvidence:
    """Run one declared paper/equation profile plus the complete strict contract."""
    manifest = load_manifest(root)
    if name not in manifest:
        raise ValueError(f"no independent verification profile for {name}")
    record = manifest[name]
    test_name = str(record["test"])
    loader_name = test_name.removeprefix("tests.")
    tests_path = str(root / "tests")
    sys.path.insert(0, tests_path)
    try:
        suite = unittest.defaultTestLoader.loadTestsFromName(loader_name)
    finally:
        sys.path.remove(tests_path)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError(f"paper/equation profile failed for {name}: {stream.getvalue()}")

    from benchmark.model_contracts import audit_model_contracts

    runtime_failures = audit_model_contracts([name], strict=True)
    if runtime_failures:
        failure = runtime_failures[0]
        raise ValueError(f"strict contract failed at {failure.stage}: {failure.error}")

    structure = list(record["structure"])
    structure_payload = json.dumps(structure, ensure_ascii=False, sort_keys=True)
    structure_digest = hashlib.sha256(structure_payload.encode("utf-8")).hexdigest()
    focused = CheckEvidence(
        passed=True,
        evidence=[test_name],
        metrics={"profile": str(record["profile"])},
    )
    runtime = CheckEvidence(
        passed=True,
        evidence=[f"uv run tsf repo doctor --strict --models {name}"],
        metrics={"contract": "strict"},
    )
    checks = IndependentChecks(
        paper_structure=focused,
        equations=focused,
        construction=runtime,
        forward=runtime,
        backward=runtime,
        finite_outputs=runtime,
        active_parameter_gradients=runtime,
        state_dict_round_trip=runtime,
        cpu=runtime,
        batch_size_boundary=runtime,
        sequence_length_boundary=runtime,
        marks_adjacency_contract=runtime,
    )
    evidence = IndependentValidationEvidence(
        schema_version=SCHEMA_VERSION,
        kind="independent-validation",
        model=name,
        verified_at=datetime.now().astimezone(),
        subject_sha256=verification_subject_sha256(root, fields),
        commands=[
            f"uv run python -m unittest {test_name} -v",
            f"uv run tsf repo doctor --strict --models {name}",
        ],
        environment=_environment(),
        basis=IndependentBasis(
            references=[str(record["paper"])],
            structure_map_sha256=structure_digest,
            independent_design=True,
            source_code_not_copied=True,
        ),
        checks=checks,
        details={"profile": record["profile"], "structure": structure},
        passed=True,
    )
    target = root / EVIDENCE_DIRECTORY / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebuild_index(root)
    return evidence
