"""Contract tests for provider-neutral Agent task templates."""

from __future__ import annotations

import unittest

from benchmark.cli import main
from tsf_core.agent_tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
)


class AgentTaskTests(unittest.TestCase):
    def test_catalog_is_valid_and_contains_bounded_workflows(self) -> None:
        self.assertEqual(audit_tasks(), [])
        self.assertEqual(
            {record["name"] for record in list_tasks()},
            {
                "autoresearch",
                "catalog-expansion",
                "component-curation",
                "experiment",
                "paper-reproduction",
                "paper-to-model",
                "paper-watch",
                "repo-final-audit",
                "verification-backlog",
            },
        )

    def test_render_binds_inputs_and_preserves_boundaries(self) -> None:
        payload = render_task(
            "autoresearch",
            {"question": "Does RevIN improve PatchTST?", "max_runs": "4"},
        )
        self.assertIn("Does RevIN improve PatchTST?", payload["prompt"])
        self.assertIn("4 total runs", payload["prompt"])
        self.assertEqual(payload["budget"]["max_runs"], 12)
        self.assertEqual(payload["permissions"]["model_code"], "no-change-without-separate-authorization")
        self.assertEqual(payload["skills"], ["run-autoresearch"])

    def test_every_template_has_a_directly_renderable_demo(self) -> None:
        for record in list_tasks():
            task = load_task(record["name"])
            supplied = {
                key: "1" if "maximum" in spec else "demo"
                for key, spec in task["inputs"].items()
                if spec.get("required", False)
            }
            payload = render_task(record["name"], supplied)
            self.assertTrue(payload["prompt"])
            self.assertEqual(payload["task"], record["name"])
            self.assertTrue(payload["permissions"])
            self.assertTrue(payload["budget"])

    def test_missing_or_unknown_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(AgentTaskError, "missing required"):
            render_task("paper-to-model", {"paper_url": "https://arxiv.org/abs/1"})
        with self.assertRaisesRegex(AgentTaskError, "unknown input"):
            render_task("paper-watch", {"surprise": "write everything"})
        with self.assertRaisesRegex(AgentTaskError, "between 1 and 12"):
            render_task("autoresearch", {"question": "test", "max_runs": "13"})

    def test_cli_routes_task_validation(self) -> None:
        self.assertEqual(main(["agent", "task", "validate"]), 0)


if __name__ == "__main__":
    unittest.main()
