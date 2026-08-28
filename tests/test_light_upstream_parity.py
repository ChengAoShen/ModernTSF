"""Regression checks for CATS and SegRNN pinned-source parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import (
    DEFAULT_INDEX,
    load_verification_index,
    verification_state,
)
from scripts.verify_light_upstream_parity import SOURCES


ROOT = Path(__file__).resolve().parents[1]


class LightUpstreamParityTests(unittest.TestCase):
    def test_exact_pinned_evidence_is_complete(self) -> None:
        for name, source in SOURCES.items():
            with self.subTest(model=name):
                recorded = json.loads(
                    (ROOT / "verification" / "parity" / f"{name}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(recorded["passed"])
                self.assertEqual(
                    recorded["upstream_execution"], "exact-pinned-checkout"
                )
                self.assertEqual(recorded["source"]["revision"], source["revision"])
                self.assertEqual(recorded["source"]["license"], source["license"])
                self.assertEqual(recorded["source"]["file_sha256"], source["sha256"])
                for case in recorded["cases"].values():
                    self.assertTrue(case["passed"])
                    self.assertTrue(case["serialization"]["local"]["passed"])
                    self.assertTrue(case["serialization"]["upstream"]["passed"])
                    self.assertTrue(case["optional_inputs"]["passed"])
                    self.assertEqual(case["optional_inputs"]["max_abs"], 0.0)
                    for mode in case["report"]["modes"].values():
                        self.assertTrue(mode["passed"])
                        self.assertTrue(mode["intermediates"])
                        self.assertTrue(mode["input_gradients"])
                        self.assertTrue(mode["parameter_gradients"])

    def test_canonical_results_are_current_and_valid(self) -> None:
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
