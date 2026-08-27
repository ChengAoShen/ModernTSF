"""Evidence-contract tests for pinned N-BEATS and N-HiTS parity."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import DEFAULT_INDEX, load_verification_index, verification_state
from scripts.verify_basis_upstream_parity import SOURCES


ROOT = Path(__file__).resolve().parents[1]


class BasisUpstreamParityTests(unittest.TestCase):
    def test_recorded_evidence_is_strict_and_source_bound(self) -> None:
        for name, source in SOURCES.items():
            with self.subTest(model=name):
                record = json.loads(
                    (ROOT / "verification" / "parity" / f"{name}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(record["passed"])
                self.assertEqual(record["upstream_execution"], "exact-pinned-source")
                self.assertEqual(record["source"]["revision"], source["revision"])
                self.assertEqual(record["source"]["license"], source["license"])
                self.assertEqual(record["source"]["sha256"], source["sha256"])
                self.assertEqual(
                    record["source"]["license_sha256"], source["license_sha256"]
                )
                for case in record["cases"].values():
                    self.assertTrue(case["passed"])
                    for mode in case["report"]["modes"].values():
                        self.assertTrue(mode["passed"])
                        self.assertTrue(mode["intermediates"])
                        self.assertTrue(mode["input_gradients"])
                        self.assertTrue(mode["parameter_gradients"])
                    self.assertTrue(case["serialization"]["local"]["passed"])
                    self.assertTrue(case["serialization"]["upstream"]["passed"])

    def test_canonical_results_are_current(self) -> None:
        snapshot = load_verification_index(ROOT / DEFAULT_INDEX)
        self.assertIsNone(snapshot.index_error)
        self.assertEqual(snapshot.errors, {})
        fields = {item["name"]: item for item in model_records(ROOT)}
        for name in SOURCES:
            with self.subTest(model=name):
                state, blockers = verification_state(ROOT, fields[name], snapshot)
                self.assertEqual(state["status"], "passed")
                self.assertEqual(blockers, [])


if __name__ == "__main__":
    unittest.main()
