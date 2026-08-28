"""Contract tests for exact-checkout BasicTS graph parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import DEFAULT_INDEX, load_verification_index, verification_state


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("AGCRN", "STGCN")


class BasicTSGraphParityTests(unittest.TestCase):
    def test_detailed_evidence_is_complete(self) -> None:
        for name in MODELS:
            with self.subTest(model=name):
                record = json.loads((ROOT / "verification" / "parity" / f"{name}.json").read_text())
                self.assertTrue(record["passed"])
                self.assertEqual(record["upstream_execution"], "exact-pinned-checkout")
                self.assertEqual(set(record["cases"]), {"batch_one_identity", "batch_two_nontrivial_graph"})
                for case in record["cases"].values():
                    self.assertTrue(case["passed"])
                    self.assertEqual(case["preprocessing_max_abs"], 0.0)
                    self.assertEqual(case["buffer_contract"]["max_abs"], 0.0)
                    for serialization in case["serialization"].values():
                        self.assertTrue(serialization[0])
                    for mode in case["report"]["modes"].values():
                        self.assertTrue(mode["passed"])
                        self.assertTrue(mode["intermediates"])
                        self.assertTrue(mode["input_gradients"])
                        self.assertTrue(mode["parameter_gradients"])

    def test_canonical_results_are_current(self) -> None:
        snapshot = load_verification_index(ROOT / DEFAULT_INDEX)
        self.assertIsNone(snapshot.index_error)
        self.assertEqual(snapshot.errors, {})
        fields = {item["name"]: item for item in model_records(ROOT)}
        for name in MODELS:
            with self.subTest(model=name):
                state, blockers = verification_state(ROOT, fields[name], snapshot)
                self.assertEqual(state["status"], "passed")
                self.assertEqual(blockers, [])


if __name__ == "__main__":
    unittest.main()
