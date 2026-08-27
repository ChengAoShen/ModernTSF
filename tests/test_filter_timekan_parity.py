"""Regression checks for committed FilterNet and TimeKAN parity evidence."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FilterTimeKANParityEvidenceTests(unittest.TestCase):
    def test_exact_checkout_details_cover_every_required_case(self):
        expected_cases = {
            "PaiFilter": {"primary", "minimum", "alternate_length"},
            "TexFilter": {"primary", "minimum", "alternate_length"},
            "TimeKAN": {"primary", "minimum_downsample", "alternate_length"},
        }
        for name, case_names in expected_cases.items():
            with self.subTest(model=name):
                path = ROOT / "verification" / "parity" / f"{name}.json"
                detail = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(detail["passed"])
                self.assertEqual(detail["upstream_execution"], "exact-pinned-checkout")
                self.assertEqual(set(detail["cases"]), case_names)
                for case in detail["cases"].values():
                    self.assertTrue(case["passed"])
                    self.assertFalse(case["active_parameters"]["omitted_local"])
                    self.assertFalse(case["active_parameters"]["omitted_upstream"])
                    self.assertTrue(case["serialization"]["local"]["passed"])
                    self.assertTrue(case["serialization"]["upstream"]["passed"])
                    for mode in case["report"]["modes"].values():
                        for group in (
                            "outputs",
                            "intermediates",
                            "input_gradients",
                            "parameter_gradients",
                        ):
                            self.assertTrue(mode[group])
                            self.assertTrue(
                                all(comparison["passed"] for comparison in mode[group].values())
                            )

    def test_timekan_declares_only_the_proven_inactive_upstream_state(self):
        detail = json.loads(
            (ROOT / "verification" / "parity" / "TimeKAN.json").read_text(
                encoding="utf-8"
            )
        )
        for case in detail["cases"].values():
            self.assertEqual(
                case["upstream_inactive_state"],
                ["enc_embedding.temporal_embedding.embed.weight"],
            )

    def test_canonical_results_are_current_and_passed(self):
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from benchmark.catalog_metadata import model_records
        from benchmark.verification_results import (
            DEFAULT_INDEX,
            load_verification_index,
            verification_state,
        )

        snapshot = load_verification_index(ROOT / DEFAULT_INDEX)
        self.assertIsNone(snapshot.index_error)
        records = {record["name"]: record for record in model_records(ROOT)}
        for name in ("PaiFilter", "TexFilter", "TimeKAN"):
            with self.subTest(model=name):
                state, blockers = verification_state(ROOT, records[name], snapshot)
                self.assertEqual(state["status"], "passed")
                self.assertEqual(blockers, [])


if __name__ == "__main__":
    unittest.main()
