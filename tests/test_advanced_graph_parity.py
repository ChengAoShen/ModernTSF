"""Contracts for exact-source D2STGNN, DFDGCN, and HimNet parity."""

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


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("D2STGNN", "DFDGCN", "HimNet")


class AdvancedGraphParityTests(unittest.TestCase):
    def test_detailed_evidence_covers_the_strict_contract(self) -> None:
        for name in MODELS:
            with self.subTest(model=name):
                record = json.loads(
                    (ROOT / "verification" / "parity" / f"{name}.json").read_text()
                )
                self.assertTrue(record["passed"])
                self.assertEqual(record["upstream_execution"], "exact-pinned-checkout")
                self.assertEqual(
                    set(record["cases"]),
                    {"batch_one_identity", "batch_two_nontrivial_graph"},
                )
                self.assertEqual(len(record["source"]["revision"]), 40)
                self.assertIn(record["source"]["license"], {"Apache-2.0", "MIT"})
                self.assertEqual(len(record["source"]["license_sha256"]), 64)
                self.assertTrue(record["source"]["files"])
                for case in record["cases"].values():
                    self.assertTrue(case["passed"])
                    self.assertEqual(case["preprocessing"]["max_abs"], 0.0)
                    for serialization in case["serialization"].values():
                        self.assertTrue(serialization[0])
                    self.assertEqual(
                        case["active_parameter_gradients"],
                        case["expected_parameter_gradients"],
                    )
                    for activity in case["gradient_activity"]["modes"].values():
                        self.assertTrue(activity["matched"])
                        self.assertEqual(activity["local"], activity["upstream"])
                    for mode in case["report"]["modes"].values():
                        self.assertTrue(mode["passed"])
                        self.assertTrue(mode["intermediates"])
                        self.assertTrue(mode["input_gradients"])
                        self.assertEqual(
                            len(mode["parameter_gradients"]),
                            case["expected_parameter_gradients"],
                        )

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
