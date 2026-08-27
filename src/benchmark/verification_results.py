"""Strict, auditable results for upstream parity and clean-room rewrites.

The model cards describe provenance.  This module stores executable evidence
separately and derives its validity against the current checkout.  It never
turns a declaration in descriptive metadata into a passing verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = 1
DEFAULT_INDEX = Path("verification/model-results.json")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


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


class SourceIdentity(_StrictModel):
    url: str
    revision: str
    license: str

    @field_validator("url")
    @classmethod
    def _absolute_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source URL must be an absolute HTTP(S) URL")
        return value

    @field_validator("revision", "license")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip() or value == "NOASSERTION":
            raise ValueError("revision and license must be explicit")
        return value


class StateMapping(_StrictModel):
    version: str
    parameters: int = Field(ge=0)
    buffers: int = Field(ge=0)

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("mapping version must not be empty")
        return value


class InputFixture(_StrictModel):
    identifier: str
    description: str

    @field_validator("identifier", "description")
    @classmethod
    def _fixture_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fixture fields must not be empty")
        return value


class Tolerances(_StrictModel):
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)


class ExecutionEnvironment(_StrictModel):
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


class UpstreamParityChecks(_StrictModel):
    outputs: CheckEvidence
    intermediates: CheckEvidence
    input_gradients: CheckEvidence
    parameter_gradients: CheckEvidence
    train_eval: CheckEvidence
    buffers: CheckEvidence
    serialization: CheckEvidence
    preprocessing: CheckEvidence
    boundaries: CheckEvidence

    @model_validator(mode="after")
    def _required_metrics(self) -> "UpstreamParityChecks":
        for name in (
            "outputs",
            "intermediates",
            "input_gradients",
            "parameter_gradients",
        ):
            metrics = getattr(self, name).metrics
            for metric in ("max_abs", "max_rel"):
                value = metrics.get(metric)
                if not isinstance(value, (float, int)) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{name} must record non-negative {metric}")
        required = {
            "train_eval": "modes",
            "buffers": "mapped_buffers",
            "serialization": "max_abs",
            "preprocessing": "contract",
            "boundaries": "cases",
        }
        for check_name, metric in required.items():
            if metric not in getattr(self, check_name).metrics:
                raise ValueError(f"{check_name} must record {metric}")
        return self


class RewriteBasis(_StrictModel):
    references: list[str] = Field(min_length=1)
    structure_map_sha256: str = Field(pattern=_SHA256_PATTERN)
    independent_design: bool
    source_code_not_copied: bool

    @field_validator("references")
    @classmethod
    def _references(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("rewrite references must not be empty")
        return values


class RewriteValidationChecks(_StrictModel):
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


class _VerificationResult(_StrictModel):
    schema_version: Literal[1]
    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
    verified_at: datetime
    subject_sha256: str = Field(pattern=_SHA256_PATTERN)
    commands: list[str] = Field(min_length=1)
    environment: ExecutionEnvironment
    artifacts: dict[str, str] = Field(min_length=1)
    passed: bool

    @field_validator("verified_at", mode="before")
    @classmethod
    def _verified_at(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("verified_at must be an ISO-8601 timestamp") from exc
        return value

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

    @field_validator("artifacts")
    @classmethod
    def _artifacts(cls, values: dict[str, str]) -> dict[str, str]:
        for relative, digest in values.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not relative.strip():
                raise ValueError("artifact paths must be safe repository-relative paths")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("artifact values must be lowercase SHA-256 digests")
        return values


class UpstreamParityResult(_VerificationResult):
    kind: Literal["upstream-parity"]
    implementation: Literal["upstream"]
    source: SourceIdentity
    mapping: StateMapping
    fixture: InputFixture
    tolerances: Tolerances
    modes: list[Literal["eval", "train"]]
    checks: UpstreamParityChecks

    @field_validator("modes")
    @classmethod
    def _both_modes(cls, values: list[str]) -> list[str]:
        if len(values) != 2 or set(values) != {"eval", "train"}:
            raise ValueError("parity evidence must cover eval and train exactly once")
        return values

    @model_validator(mode="after")
    def _consistent_pass(self) -> "UpstreamParityResult":
        checks_passed = all(
            check.passed for check in self.checks.__dict__.values()
        )
        if self.passed != checks_passed:
            raise ValueError("passed must equal the conjunction of all parity checks")
        return self


class RewriteValidationResult(_VerificationResult):
    kind: Literal["rewrite-validation"]
    implementation: Literal["rewrite"]
    basis: RewriteBasis
    checks: RewriteValidationChecks

    @model_validator(mode="after")
    def _consistent_pass(self) -> "RewriteValidationResult":
        checks_passed = all(
            check.passed for check in self.checks.__dict__.values()
        )
        basis_passed = self.basis.independent_design and self.basis.source_code_not_copied
        if self.passed != (checks_passed and basis_passed):
            raise ValueError(
                "passed must require the clean-room basis and every rewrite check"
            )
        return self


VerificationResult = Annotated[
    UpstreamParityResult | RewriteValidationResult, Field(discriminator="kind")
]


class VerificationIndex(_StrictModel):
    schema_version: Literal[1]
    results: dict[str, VerificationResult]

    @model_validator(mode="after")
    def _matching_keys(self) -> "VerificationIndex":
        mismatches = [key for key, result in self.results.items() if key != result.model]
        if mismatches:
            raise ValueError(f"result keys do not match model names: {mismatches}")
        return self


@dataclass(frozen=True)
class VerificationSnapshot:
    """Usable records plus parse errors retained for per-model audit output."""

    results: dict[str, UpstreamParityResult | RewriteValidationResult]
    errors: dict[str, str]
    index_error: str | None = None


@contextmanager
def _index_lock(path: Path):
    """Serialize read-modify-replace updates when batch workers finish together."""
    lock_name = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"moderntsf-verification-{lock_name}.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - atomic replacement remains safe
            yield
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def load_verification_index(path: Path) -> VerificationSnapshot:
    """Read an index without hiding malformed per-model evidence."""
    if not path.is_file():
        return VerificationSnapshot({}, {}, "missing")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return VerificationSnapshot({}, {}, f"invalid JSON: {exc}")
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "results"}:
        return VerificationSnapshot({}, {}, "invalid index envelope")
    if raw["schema_version"] != SCHEMA_VERSION or not isinstance(raw["results"], dict):
        return VerificationSnapshot({}, {}, "unsupported schema or invalid results map")

    from pydantic import TypeAdapter, ValidationError

    adapter = TypeAdapter(VerificationResult)
    results: dict[str, UpstreamParityResult | RewriteValidationResult] = {}
    errors: dict[str, str] = {}
    for name, payload in raw["results"].items():
        if not isinstance(name, str):
            errors[str(name)] = "model key must be a string"
            continue
        try:
            result = adapter.validate_python(payload)
            if result.model != name:
                raise ValueError("result model does not match its index key")
            results[name] = result
        except (ValidationError, ValueError) as exc:
            errors[name] = str(exc)
    return VerificationSnapshot(results, errors)


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

    from components.catalog import COMPONENT_CATALOG

    for name in fields.get("components", ()):
        module = COMPONENT_CATALOG.get(str(name)).module
        paths.add(root / "src" / Path(*module.split(".")).with_suffix(".py"))
    adapter_name = fields.get("adapter")
    if adapter_name:
        from adapters.catalog import ADAPTER_CATALOG

        module = ADAPTER_CATALOG[str(adapter_name)].module
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
    """Return the SHA-256 digest used to bind an index record to an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verification_state(
    root: Path,
    fields: dict[str, object],
    snapshot: VerificationSnapshot,
) -> tuple[dict[str, object], list[str]]:
    """Return display state and computed blockers for one current model."""
    implementation = str(fields.get("implementation", ""))
    prefix = "upstream.parity" if implementation == "upstream" else "rewrite.validation"
    if snapshot.index_error:
        state = "index-missing" if snapshot.index_error == "missing" else "index-invalid"
        return {"status": state, "detail": snapshot.index_error}, [f"{prefix}.{state}"]
    name = str(fields["name"])
    if name in snapshot.errors:
        return {"status": "invalid", "detail": snapshot.errors[name]}, [f"{prefix}.invalid"]
    result = snapshot.results.get(name)
    if result is None:
        return {"status": "missing"}, [f"{prefix}.missing"]
    expected_kind = "upstream-parity" if implementation == "upstream" else "rewrite-validation"
    if result.kind != expected_kind or result.implementation != implementation:
        return {"status": "invalid", "detail": "implementation route mismatch"}, [f"{prefix}.invalid"]
    expected_digest = verification_subject_sha256(root, fields)
    stale = result.subject_sha256 != expected_digest
    for relative, expected_artifact_digest in result.artifacts.items():
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError:
            stale = True
            continue
        if not artifact.is_file() or evidence_file_sha256(artifact) != expected_artifact_digest:
            stale = True
    if isinstance(result, UpstreamParityResult):
        codebase = dict(fields.get("codebase", {}))
        stale = stale or any(
            getattr(result.source, key) != codebase.get(key)
            for key in ("url", "revision", "license")
        )
    common = {
        "kind": result.kind,
        "verified_at": result.verified_at.isoformat(),
        "status": "stale" if stale else "passed" if result.passed else "failed",
    }
    if stale:
        return common, [f"{prefix}.stale"]
    if not result.passed:
        return common, [f"{prefix}.failed"]
    return common, []


def write_verification_result(
    path: Path,
    result: UpstreamParityResult | RewriteValidationResult | dict[str, object],
) -> None:
    """Validate and atomically add or replace one result in a healthy index."""
    from pydantic import TypeAdapter

    validated = TypeAdapter(VerificationResult).validate_python(result)
    root = path.parent.parent.resolve()
    for relative, expected in validated.artifacts.items():
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact escapes repository: {relative}") from exc
        if not artifact.is_file() or evidence_file_sha256(artifact) != expected:
            raise ValueError(f"artifact is missing or its digest differs: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _index_lock(path):
        snapshot = load_verification_index(path)
        if snapshot.index_error and snapshot.index_error != "missing":
            raise ValueError(f"refusing to overwrite invalid index: {snapshot.index_error}")
        if snapshot.errors:
            raise ValueError("refusing to overwrite index with invalid model results")
        records = {**snapshot.results, validated.model: validated}
        index = VerificationIndex(
            schema_version=SCHEMA_VERSION,
            results=records,
        )
        payload = json.dumps(index.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
