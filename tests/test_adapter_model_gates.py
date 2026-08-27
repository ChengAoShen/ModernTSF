"""Ensure approximation adapters remain visible as unfinished model work."""

from __future__ import annotations

from pathlib import Path
import unittest

from benchmark.catalog_metadata import model_records
from benchmark.commands.catalog_resources import _model_audit_record
from benchmark.verification_results import DEFAULT_INDEX, load_verification_index


ROOT = Path(__file__).resolve().parents[1]


class AdapterModelGateTests(unittest.TestCase):
    def test_approximation_adapter_blocks_named_model_completion(self) -> None:
        fields = {record["name"]: record for record in model_records(ROOT)}
        snapshot = load_verification_index(ROOT / DEFAULT_INDEX)
        amrc = _model_audit_record(fields["AMRC"], verification=snapshot)
        self.assertEqual(amrc["adapter"], "recent-tsf")
        self.assertIn("adapter.approximation", amrc["blockers"])

        rewritten = _model_audit_record(
            fields["AutoRegressiveTS"], verification=snapshot
        )
        self.assertIsNone(rewritten["adapter"])
        self.assertNotIn("adapter.approximation", rewritten["blockers"])


if __name__ == "__main__":
    unittest.main()
