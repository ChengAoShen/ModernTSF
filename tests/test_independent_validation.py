"""Contracts for route-neutral independent model verification evidence."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.independent_validation import (
    IndependentValidationEvidence,
    evidence_state,
    file_sha256,
    rebuild_index,
)
from benchmark.catalog_metadata import model_records
from tsf_core.paths import repository_root


def _check() -> dict[str, object]:
    return {"passed": True, "evidence": ["tests/test_example.py"], "metrics": {}}


class IndependentValidationTests(unittest.TestCase):
    def _payload(self, subject: str) -> dict[str, object]:
        checks = {
            name: _check()
            for name in (
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
            "kind": "independent-validation",
            "model": "Example",
            "verified_at": datetime.now(UTC),
            "subject_sha256": subject,
            "commands": ["uv run tsf verify model Example"],
            "environment": {
                "python": "3.12",
                "framework": "torch",
                "dependencies": {"torch": "2.6.0"},
                "platform": "test",
                "device": "cpu",
                "dtype": "float32",
                "deterministic": {"seed": 1},
            },
            "basis": {
                "references": ["paper"],
                "structure_map_sha256": "b" * 64,
                "independent_design": True,
                "source_code_not_copied": True,
            },
            "checks": checks,
            "details": {"structure": "paper-derived"},
            "passed": True,
        }

    def test_schema_rejects_a_false_independence_claim(self) -> None:
        payload = self._payload("a" * 64)
        payload["basis"]["source_code_not_copied"] = False
        with self.assertRaisesRegex(ValueError, "independent basis"):
            IndependentValidationEvidence.model_validate(payload)

    def test_index_is_a_generated_digest_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_dir = root / "verification" / "evidence"
            evidence_dir.mkdir(parents=True)
            evidence = evidence_dir / "Example.json"
            evidence.write_text(
                IndependentValidationEvidence.model_validate(
                    self._payload("a" * 64)
                ).model_dump_json(),
                encoding="utf-8",
            )
            index = rebuild_index(root)
            self.assertEqual(set(index.models), {"Example"})
            self.assertEqual(index.models["Example"].sha256, file_sha256(evidence))
            saved = json.loads((root / "verification" / "index.json").read_text())
            self.assertEqual(saved["models"]["Example"]["evidence"], "verification/evidence/Example.json")

    def test_missing_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = evidence_state(Path(temporary), "Example", {})
            self.assertEqual(state.status, "missing")

    def test_migration_index_contains_only_existing_independent_models(self) -> None:
        root = repository_root()
        fields = {str(record["name"]): record for record in model_records(root)}
        current_independent = {
            name for name, record in fields.items() if record["implementation"] == "rewrite"
        }
        index = json.loads((root / "verification" / "index.json").read_text())
        self.assertEqual(set(index["models"]), current_independent)
        states = [evidence_state(root, name, fields[name]).status for name in current_independent]
        self.assertEqual(set(states), {"passed"})


if __name__ == "__main__":
    unittest.main()
