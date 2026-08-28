from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.verification_results import (
    RewriteValidationResult,
    UpstreamParityResult,
    evidence_file_sha256,
    load_verification_index,
    verification_state,
    verification_subject_sha256,
    write_verification_result,
)


def _check(passed: bool = True) -> dict[str, object]:
    return {"passed": passed, "evidence": ["uv run pytest focused-test"]}


def _environment() -> dict[str, object]:
    return {
        "python": "3.12",
        "framework": "torch 2.6.0",
        "dependencies": {"torch": "2.6.0"},
        "platform": "test-platform",
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"seed": 0, "algorithms": True},
    }


def _upstream_result(name: str, digest: str, artifact_digest: str) -> dict[str, object]:
    checks = {
        key: _check()
        for key in (
            "outputs",
            "intermediates",
            "input_gradients",
            "parameter_gradients",
            "train_eval",
            "buffers",
            "serialization",
            "preprocessing",
            "boundaries",
        )
    }
    for check_name in (
        "outputs",
        "intermediates",
        "input_gradients",
        "parameter_gradients",
    ):
        checks[check_name]["metrics"] = {"max_abs": 0.0, "max_rel": 0.0}
    checks["train_eval"]["metrics"] = {"modes": "eval,train"}
    checks["buffers"]["metrics"] = {"mapped_buffers": 0}
    checks["serialization"]["metrics"] = {"max_abs": 0.0}
    checks["preprocessing"]["metrics"] = {"contract": "identity"}
    checks["boundaries"]["metrics"] = {"cases": "minimal"}
    return {
        "schema_version": 1,
        "kind": "upstream-parity",
        "model": name,
        "implementation": "upstream",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": digest,
        "commands": ["uv run python parity_case.py"],
        "environment": _environment(),
        "artifacts": {"verification/raw/Example.json": artifact_digest},
        "passed": True,
        "source": {
            "url": "https://example.com/upstream",
            "revision": "0123456789abcdef",
            "license": "MIT",
        },
        "mapping": {"version": "v1", "parameters": 2, "buffers": 0},
        "fixture": {"identifier": "minimal-v1", "description": "B=2,L=8,C=3"},
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "modes": ["eval", "train"],
        "checks": checks,
    }


def _rewrite_result(name: str, digest: str, artifact_digest: str) -> dict[str, object]:
    checks = {
        key: _check()
        for key in (
            "paper_structure",
            "equations",
            "construction",
            "forward",
            "backward",
            "finite_outputs",
            "active_parameter_gradients",
            "state_dict_round_trip",
            "cpu",
            "batch_size_boundary",
            "sequence_length_boundary",
            "marks_adjacency_contract",
        )
    }
    return {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": digest,
        "commands": ["uv run pytest rewrite_case.py"],
        "environment": _environment(),
        "artifacts": {"verification/raw/Example.json": artifact_digest},
        "passed": True,
        "basis": {
            "references": ["https://example.com/paper"],
            "structure_map_sha256": "a" * 64,
            "independent_design": True,
            "source_code_not_copied": True,
        },
        "checks": checks,
    }


class VerificationResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        package = self.root / "src/models/example"
        package.mkdir(parents=True)
        (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        (package / "README.md").write_text("model card\n", encoding="utf-8")
        config = self.root / "configs/models/example.toml"
        config.parent.mkdir(parents=True)
        config.write_text("[model]\n", encoding="utf-8")
        self.fields: dict[str, object] = {
            "name": "Example",
            "package": "example",
            "implementation": "upstream",
            "config_path": "configs/models/example.toml",
            "components": (),
            "codebase": {
                "url": "https://example.com/upstream",
                "revision": "0123456789abcdef",
                "license": "MIT",
            },
        }
        self.index = self.root / "verification/model-results.json"
        self.artifact = self.root / "verification/raw/Example.json"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_text('{"passed": true}\n', encoding="utf-8")
        self.artifact_digest = evidence_file_sha256(self.artifact)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_atomic_round_trip_and_passing_upstream_evidence(self) -> None:
        digest = verification_subject_sha256(self.root, self.fields)
        write_verification_result(
            self.index, _upstream_result("Example", digest, self.artifact_digest)
        )
        snapshot = load_verification_index(self.index)
        self.assertEqual(snapshot.errors, {})
        self.assertIsInstance(snapshot.results["Example"], UpstreamParityResult)
        state, blockers = verification_state(self.root, self.fields, snapshot)
        self.assertEqual(state["status"], "passed")
        self.assertEqual(blockers, [])
        self.assertEqual(
            json.loads(self.index.read_text(encoding="utf-8"))["schema_version"], 1
        )

    def test_missing_invalid_and_stale_results_are_distinct(self) -> None:
        self.index.parent.mkdir(parents=True, exist_ok=True)
        self.index.write_text('{"schema_version": 1, "results": {}}\n', encoding="utf-8")
        state, blockers = verification_state(
            self.root, self.fields, load_verification_index(self.index)
        )
        self.assertEqual((state["status"], blockers), ("missing", ["upstream.parity.missing"]))

        raw = _upstream_result(
            "Example", verification_subject_sha256(self.root, self.fields), self.artifact_digest
        )
        raw["unexpected"] = True
        self.index.write_text(
            json.dumps({"schema_version": 1, "results": {"Example": raw}}, default=str),
            encoding="utf-8",
        )
        state, blockers = verification_state(
            self.root, self.fields, load_verification_index(self.index)
        )
        self.assertEqual(state["status"], "invalid")
        self.assertEqual(blockers, ["upstream.parity.invalid"])

        write_payload = _upstream_result(
            "Example", verification_subject_sha256(self.root, self.fields), self.artifact_digest
        )
        self.index.unlink()
        write_verification_result(self.index, write_payload)
        self.artifact.write_text('{"passed": false}\n', encoding="utf-8")
        state, blockers = verification_state(
            self.root, self.fields, load_verification_index(self.index)
        )
        self.assertEqual(state["status"], "stale")
        self.assertEqual(blockers, ["upstream.parity.stale"])
        self.artifact.write_text('{"passed": true}\n', encoding="utf-8")
        (self.root / "src/models/example/model.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        state, blockers = verification_state(
            self.root, self.fields, load_verification_index(self.index)
        )
        self.assertEqual(state["status"], "stale")
        self.assertEqual(blockers, ["upstream.parity.stale"])

    def test_subject_hash_binds_transitive_component_dependencies(self) -> None:
        components = self.root / "src/components"
        components.mkdir(parents=True)
        (components / "patchtst.py").write_text(
            "from components.revin import RevIN\n", encoding="utf-8"
        )
        dependency = components / "revin.py"
        dependency.write_text("VALUE = 1\n", encoding="utf-8")
        self.fields["components"] = ("patchtst",)

        with patch("components.audit.COMPONENTS", components):
            original = verification_subject_sha256(self.root, self.fields)
            dependency.write_text("VALUE = 2\n", encoding="utf-8")
            changed = verification_subject_sha256(self.root, self.fields)

        self.assertNotEqual(original, changed)

    def test_rewrite_schema_and_route_are_strict(self) -> None:
        self.fields["implementation"] = "rewrite"
        self.fields["codebase"] = {
            "url": "",
            "revision": "",
            "license": "",
        }
        digest = verification_subject_sha256(self.root, self.fields)
        write_verification_result(
            self.index, _rewrite_result("Example", digest, self.artifact_digest)
        )
        snapshot = load_verification_index(self.index)
        self.assertIsInstance(snapshot.results["Example"], RewriteValidationResult)
        self.assertEqual(verification_state(self.root, self.fields, snapshot)[1], [])

        malformed = _rewrite_result("Example", digest, self.artifact_digest)
        del malformed["checks"]["equations"]
        self.index.unlink()
        with self.assertRaisesRegex(Exception, "equations"):
            write_verification_result(self.index, malformed)

    def test_writer_refuses_to_destroy_an_invalid_index(self) -> None:
        self.index.parent.mkdir(parents=True, exist_ok=True)
        self.index.write_text("not json", encoding="utf-8")
        digest = verification_subject_sha256(self.root, self.fields)
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            write_verification_result(
                self.index, _upstream_result("Example", digest, self.artifact_digest)
            )

    def test_parallel_writers_do_not_lose_results(self) -> None:
        digest = verification_subject_sha256(self.root, self.fields)
        records = [
            _upstream_result(name, digest, self.artifact_digest)
            for name in ("ExampleA", "ExampleB", "ExampleC")
        ]
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(
                executor.map(
                    lambda record: write_verification_result(self.index, record),
                    records,
                )
            )
        snapshot = load_verification_index(self.index)
        self.assertEqual(set(snapshot.results), {"ExampleA", "ExampleB", "ExampleC"})


if __name__ == "__main__":
    unittest.main()
