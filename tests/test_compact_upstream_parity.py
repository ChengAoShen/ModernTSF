"""Pinned numerical parity tests for compact upstream forecasting models."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch
import torch.nn as nn

from benchmark.catalog_metadata import model_records
from benchmark.parity import compare_model_parity
from benchmark.verification_results import (
    DEFAULT_INDEX,
    load_verification_index,
    verification_state,
)
from scripts.verify_compact_upstream_parity import SOURCES, verify_model


ROOT = Path(__file__).resolve().parents[1]


class CompactUpstreamParityTests(unittest.TestCase):
    def test_complex_comparison_includes_imaginary_component(self) -> None:
        class Identity(nn.Module):
            def forward(self, value):
                return value

        class Conjugate(nn.Module):
            def forward(self, value):
                return value.conj()

        value = torch.tensor([1.0 + 2.0j])
        report = compare_model_parity(
            Identity(),
            Conjugate(),
            (value,),
            compare_gradients=False,
        )
        self.assertFalse(report.passed)

    def test_checked_evidence_matches_executable_fixtures(self) -> None:
        for name in SOURCES:
            with self.subTest(model=name):
                actual = verify_model(name)
                self.assertTrue(actual["passed"])
                recorded = json.loads(
                    (ROOT / "verification" / "parity" / f"{name}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(recorded["passed"])
                self.assertEqual(recorded["upstream_execution"], "exact-pinned-checkout")
                self.assertEqual(recorded["source"], actual["source"])
                self.assertEqual(recorded["mapping_version"], actual["mapping_version"])
                for case in recorded["cases"].values():
                    self.assertTrue(case["passed"])
                    self.assertTrue(case["serialization"]["local"]["passed"])
                    self.assertTrue(case["serialization"]["upstream"]["passed"])
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
