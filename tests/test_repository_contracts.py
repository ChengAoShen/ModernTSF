"""Focused regression tests for the Agent-first catalog layers."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapters.audit import audit_adapters
from benchmark.command_runtime import module_slug as cli_module_slug
from benchmark.cli import main as cli_main
from benchmark.commands.check_registry import check as check_model_catalog
from benchmark.model_contracts import audit_model_contracts
from benchmark.commands.new_model import _module_slug as scaffold_module_slug
from benchmark.runner.model_io import call_forecaster, slice_prediction_target
from components.adj_norm import gcn_norm, transition_matrix
from components.audit import audit_components
from components.catalog import COMPONENT_CATALOG
from components.flatten_forecast_head import FlattenForecastHead
from components.graph_utils import adj_to_supports
from components.marks import to_spatiotemporal
from components.quantile_head import QuantileHead
from components.revin import RevIN
from components.series_decomposition import (
    EdgePaddedMovingAverage,
    SeriesDecomposition,
)
from tsf_core.agent_assets import audit_agent_assets


class RepositoryContractTests(unittest.TestCase):
    def test_python_modules_have_descriptions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        missing = []
        for path in (root / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not ast.get_docstring(tree):
                missing.append(str(path.relative_to(root)))
        self.assertEqual(missing, [])

    def test_public_module_slug_is_consistent(self) -> None:
        for normalize in (cli_module_slug, scaffold_module_slug):
            self.assertEqual(normalize("AirFormer"), "airformer")
            self.assertEqual(normalize("S_Mamba"), "s_mamba")

    def test_cli_routes_lightweight_catalog_descriptions(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["component", "show", "quantile_head"]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["module"], "components.quantile_head")
        self.assertIn("quantile_dlinear", payload["consumers"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["model", "list", "--json"]), 0)
        records = json.loads(output.getvalue())
        self.assertEqual(len(records), 178)
        self.assertTrue(all(record["summary"] for record in records))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(
                    [
                        "component",
                        "match",
                        "patch",
                        "transformer",
                        "backbone",
                        "--json",
                    ]
                ),
                0,
            )
        matches = json.loads(output.getvalue())
        self.assertEqual(matches[0]["name"], "patchtst")
        self.assertTrue(matches[0]["review_required"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli_main(["model", "audit", "--summary"]), 1)
        audit = json.loads(output.getvalue())
        self.assertEqual(audit["models"], 178)
        self.assertEqual(audit["reviewed"], 178)
        self.assertEqual(audit["unreviewed"], 0)
        self.assertEqual(sum(audit["evidence"].values()), 178)
        self.assertEqual(
            sum(audit["incomplete_by_evidence"].values()), audit["incomplete"]
        )
        self.assertEqual(audit["blockers"]["verified evidence"], 42)
        self.assertIn("complete_source", audit)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(["model", "audit", "CATS", "BiST", "--json"]), 1
            )
        records = {record["name"]: record for record in json.loads(output.getvalue())}
        self.assertTrue(records["CATS"]["complete"])
        self.assertTrue(records["BiST"]["reviewed"])
        self.assertEqual(records["CATS"]["source"]["missing"], [])
        self.assertIn("license", records["BiST"]["source"]["missing"])

    def test_agent_assets_are_canonical(self) -> None:
        self.assertEqual(audit_agent_assets(), [])

    def test_model_and_component_catalogs_are_consistent(self) -> None:
        self.assertEqual(check_model_catalog(), [])
        self.assertEqual(audit_adapters(), [])
        self.assertEqual(audit_components(), [])

    def test_selected_model_contract_batch(self) -> None:
        self.assertEqual(
            audit_model_contracts(
                ["ETSformer", "MixLinear", "CRIB"], forward=True
            ),
            [],
        )

    def test_shared_adjacency_normalizers_are_finite(self) -> None:
        adjacency = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(gcn_norm(adjacency)).all())
        self.assertTrue(np.isfinite(transition_matrix(adjacency)).all())
        forward, reverse = adj_to_supports(adjacency)
        self.assertEqual(forward.dtype, torch.float32)
        torch.testing.assert_close(
            forward, torch.from_numpy(transition_matrix(adjacency)).float()
        )
        torch.testing.assert_close(
            reverse, torch.from_numpy(transition_matrix(adjacency.T)).float()
        )

    def test_shared_spatiotemporal_adapter_shape(self) -> None:
        values = torch.randn(2, 12, 4)
        marks = torch.zeros(2, 12, 6)
        adapted = to_spatiotemporal(values, marks)
        self.assertEqual(adapted.shape, (2, 12, 4, 3))
        torch.testing.assert_close(adapted[..., 0], values)

    def test_revin_round_trip(self) -> None:
        values = torch.randn(2, 24, 3)
        revin = RevIN(3)
        restored = revin(revin(values, "norm"), "denorm")
        torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)

    def test_edge_padded_series_decomposition_matches_reference(self) -> None:
        values = torch.randn(2, 9, 3, requires_grad=True)
        smoother = EdgePaddedMovingAverage(3)
        actual = smoother(values)
        padded = torch.cat([values[:, :1], values, values[:, -1:]], dim=1)
        expected = F.avg_pool1d(padded.permute(0, 2, 1), 3, stride=1).permute(
            0, 2, 1
        )
        torch.testing.assert_close(actual, expected)
        residual, trend = SeriesDecomposition(3)(values)
        torch.testing.assert_close(residual + trend, values)
        (residual.square().mean() + trend.square().mean()).backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        with self.assertRaisesRegex(ValueError, "positive odd"):
            EdgePaddedMovingAverage(4)

    def test_flatten_forecast_head_shared_and_individual_contracts(self) -> None:
        values = torch.randn(2, 3, 4, 5, requires_grad=True)
        shared = FlattenForecastHead(False, 3, 20, 7)
        shared_output = shared(values)
        torch.testing.assert_close(
            shared_output, shared.linear(values.flatten(start_dim=-2))
        )
        individual = FlattenForecastHead(True, 3, 20, 7)
        individual_output = individual(values)
        expected = torch.stack(
            [
                individual.linears[index](values[:, index].flatten(start_dim=-2))
                for index in range(3)
            ],
            dim=1,
        )
        torch.testing.assert_close(individual_output, expected)
        self.assertEqual(shared_output.shape, (2, 3, 7))
        self.assertEqual(individual_output.shape, (2, 3, 7))
        (shared_output.mean() + individual_output.mean()).backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_extracted_component_catalog_surface(self) -> None:
        for name in (
            "dlinear",
            "flatten_forecast_head",
            "mamba",
            "patchtst",
            "quantile_head",
            "series_decomposition",
        ):
            self.assertIn(name, COMPONENT_CATALOG.names())
            self.assertTrue(COMPONENT_CATALOG.get(name).contract)

    def test_quantile_head_is_monotone_and_differentiable(self) -> None:
        values = torch.randn(2, 8, 3, 4, requires_grad=True)
        head = QuantileHead([0.1, 0.5, 0.9], in_features=4)
        output = head(values)
        self.assertEqual(output.shape, (2, 8, 3, 3))
        self.assertTrue(torch.all(output[..., 1:] >= output[..., :-1]))
        output.mean().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_model_io_preserves_probabilistic_axis(self) -> None:
        output = torch.randn(2, 12, 4, 9)
        target = torch.randn(2, 16, 4)
        sliced_output, sliced_target = slice_prediction_target(output, target, 6, "MS")
        self.assertEqual(sliced_output.shape, (2, 6, 1, 9))
        self.assertEqual(sliced_target.shape, (2, 6, 1))

    def test_model_io_does_not_mask_internal_type_errors(self) -> None:
        class Broken(torch.nn.Module):
            def forward(self, x):
                raise TypeError("internal failure")

        values = torch.randn(1, 4, 2)
        with self.assertRaisesRegex(TypeError, "internal failure"):
            call_forecaster(Broken(), values, None, values, None)


if __name__ == "__main__":
    unittest.main()
