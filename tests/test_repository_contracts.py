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
from benchmark.catalog_metadata import model_records
from benchmark.cli import main as cli_main
from benchmark.commands.check_registry import check as check_model_catalog
from benchmark.model_contracts import audit_model_contracts
from benchmark.parity import compare_model_parity
from benchmark.commands.new_model import _module_slug as scaffold_module_slug
from benchmark.runner.model_io import call_forecaster, slice_prediction_target
from components.adj_norm import gcn_norm, transition_matrix
from components.audit import audit_components
from components.catalog import COMPONENT_CATALOG
from components.channel_wise_linear import ChannelWiseLinear
from components.dominant_periods import dominant_periods
from components.flatten_forecast_head import FlattenForecastHead
from components.gaussian_parameter_head import GaussianParameterHead
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
                        "model",
                        "search",
                        "exogenous",
                        "transformer",
                        "--json",
                    ]
                ),
                0,
            )
        search_results = json.loads(output.getvalue())
        self.assertEqual(search_results[0]["name"], "TimeXer")
        self.assertIn("exogenous", search_results[0]["matched_terms"])

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
        self.assertEqual(audit["implementation"], {"rewrite": 147, "upstream": 31})
        self.assertEqual(
            sum(audit["failed_by_implementation"].values()), audit["failed"]
        )
        self.assertEqual(audit["blockers"]["upstream.parity"], 31)
        self.assertEqual(audit["complete_upstream_codebase"], 31)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                cli_main(["model", "audit", "CATS", "BiST", "--json"]), 1
            )
        records = {record["name"]: record for record in json.loads(output.getvalue())}
        self.assertEqual(records["CATS"]["implementation"], "upstream")
        self.assertEqual(records["CATS"]["codebase"]["missing"], [])
        self.assertIn("upstream.parity", records["CATS"]["blockers"])
        self.assertEqual(records["BiST"]["implementation"], "rewrite")
        self.assertEqual(records["BiST"]["codebase"]["usage"], "reference-only")

    def test_model_cards_are_the_only_descriptive_metadata_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        records = model_records(root)
        self.assertEqual(len(records), 178)
        self.assertEqual(
            {record["implementation"] for record in records},
            {"upstream", "rewrite"},
        )
        forbidden = {"paper", "source", "evidence", "deviations", "implementation"}
        for record in records:
            spec_path = root / str(record["spec_file"])
            tree = ast.parse(spec_path.read_text(encoding="utf-8"))
            spec_call = next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets)
            )
            self.assertTrue(forbidden.isdisjoint({kw.arg for kw in spec_call.keywords}))

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
        subtract_last = RevIN(3, subtract_last=True)
        restored = subtract_last(subtract_last(values, "norm"), "denorm")
        torch.testing.assert_close(restored, values, atol=1e-5, rtol=1e-5)
        disabled = RevIN(3, enabled=False)
        self.assertIs(disabled(values, "norm"), values)
        self.assertIs(disabled(values, "denorm"), values)

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

    def test_channel_wise_linear_matches_original_output_and_gradients(self) -> None:
        for individual in (False, True):
            actual_input = torch.randn(2, 3, 5, requires_grad=True)
            reference_input = actual_input.detach().clone().requires_grad_(True)
            projection = ChannelWiseLinear(5, 4, 3, individual)
            actual = projection(actual_input)
            if individual:
                reference = torch.zeros(2, 3, 4)
                for index in range(3):
                    reference[:, index, :] = projection.linears[index](
                        reference_input[:, index, :]
                    )
            else:
                reference = projection.linear(reference_input)
            torch.testing.assert_close(actual, reference)
            actual_grad = torch.autograd.grad(actual.square().sum(), actual_input)[0]
            reference_grad = torch.autograd.grad(
                reference.square().sum(), reference_input
            )[0]
            torch.testing.assert_close(actual_grad, reference_grad)

    def test_dominant_periods_matches_timesnet_msgnet_reference(self) -> None:
        actual_input = torch.randn(3, 16, 4, requires_grad=True)
        reference_input = actual_input.detach().clone().requires_grad_(True)
        periods, weights = dominant_periods(actual_input, 3)

        spectrum = torch.fft.rfft(reference_input, dim=1)
        strength = abs(spectrum).mean(0).mean(-1)
        strength[0] = 0
        _, indices = torch.topk(strength, 3)
        indices = indices.detach().cpu().numpy()
        reference_periods = reference_input.shape[1] // indices
        reference_weights = abs(spectrum).mean(-1)[:, indices]

        np.testing.assert_array_equal(periods, reference_periods)
        torch.testing.assert_close(weights, reference_weights)
        actual_grad = torch.autograd.grad(weights.sum(), actual_input)[0]
        reference_grad = torch.autograd.grad(
            reference_weights.sum(), reference_input
        )[0]
        torch.testing.assert_close(actual_grad, reference_grad)

    def test_gaussian_parameter_head_preserves_both_scale_formulas(self) -> None:
        values = torch.randn(2, 5, requires_grad=True)
        for transform in ("softplus", "log1pexp"):
            head = GaussianParameterHead(5, 3, eps=1e-6, scale_transform=transform)
            loc, scale = head(values)
            raw_scale = head.scale_layer(values)
            expected_scale = (
                F.softplus(raw_scale)
                if transform == "softplus"
                else torch.log(1 + torch.exp(raw_scale))
            ) + 1e-6
            torch.testing.assert_close(loc, head.loc_layer(values))
            torch.testing.assert_close(scale, expected_scale)
            self.assertTrue(torch.all(scale > 0))
            gradient = torch.autograd.grad(
                (loc + scale).sum(), values, retain_graph=True
            )[0]
            self.assertTrue(torch.isfinite(gradient).all())

    def test_extracted_component_catalog_surface(self) -> None:
        for name in (
            "dlinear",
            "channel_wise_linear",
            "dominant_periods",
            "flatten_forecast_head",
            "gaussian_parameter_head",
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

    def test_numerical_parity_harness_compares_outputs_and_gradients(self) -> None:
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(4, 3)

            def forward(self, values):
                return torch.tanh(self.projection(values))

        upstream = Block()
        local = Block()
        report = compare_model_parity(
            local,
            upstream,
            (torch.randn(2, 5, 4),),
            module_map={"projection": "projection"},
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.to_dict()["modes"]["train"]["passed"])


if __name__ == "__main__":
    unittest.main()
