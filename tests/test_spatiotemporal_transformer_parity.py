"""Contracts for STAEformer, StemGNN, and TimeBridge parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import DEFAULT_INDEX, load_verification_index, verification_state


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("STAEformer", "StemGNN", "TimeBridge")


class SpatiotemporalTransformerParityTests(unittest.TestCase):
    def test_detailed_evidence_covers_strict_contract(self) -> None:
        for name in MODELS:
            with self.subTest(model=name):
                detail = json.loads((ROOT / "verification" / "parity" / f"{name}.json").read_text())
                self.assertTrue(detail["passed"])
                self.assertEqual(detail["upstream_execution"], "exact-pinned-checkout")
                self.assertEqual(set(detail["cases"]), {"batch_one", "batch_two"})
                self.assertEqual(len(detail["source"]["revision"]), 40)
                self.assertEqual(len(detail["source"]["license_sha256"]), 64)
                self.assertTrue(detail["source"]["files"])
                for case in detail["cases"].values():
                    self.assertTrue(case["passed"])
                    self.assertEqual(case["preprocessing"]["max_abs"], 0.0)
                    self.assertEqual(case["active_parameter_gradients"], case["expected_parameter_gradients"])
                    self.assertTrue(all(item[0] for item in case["serialization"].values()))
                    self.assertTrue(all(item["matched"] for item in case["gradient_activity"]["modes"].values()))
                    for mode in case["report"]["modes"].values():
                        self.assertTrue(mode["passed"])
                        self.assertTrue(mode["intermediates"])
                        self.assertTrue(mode["input_gradients"])
                        self.assertEqual(len(mode["parameter_gradients"]), case["expected_parameter_gradients"])

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
