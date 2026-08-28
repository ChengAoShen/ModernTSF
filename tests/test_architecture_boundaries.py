"""Regression tests for catalog, profiling, and training-layer boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch import nn

from benchmark.catalog_metadata import model_records
from benchmark.config.schema.runtime import ExperimentRuntimeConfig
from benchmark.registry.losses import LOSS_NAME_MAP
from benchmark.registry.models import MODEL_CATALOG
from benchmark.runner.trainer import _forward_training


ROOT = Path(__file__).resolve().parents[1]


class _FourInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.calls.append((x_enc, x_mark_enc, x_dec, x_mark_dec))
        return x_enc[:, -2:, :]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_retired_runtime_and_loss_aliases_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentRuntimeConfig.model_validate({"gpus": [0, 1]})
        self.assertNotIn("l1", LOSS_NAME_MAP)

    def test_catalog_metadata_uses_registration_as_admission_boundary(self) -> None:
        records = model_records(ROOT, refs={"Linear": "models.linear.spec"})
        self.assertEqual([record["name"] for record in records], ["Linear"])

    def test_profiler_uses_canonical_four_input_call_and_restores_mode(self) -> None:
        profile = importlib.import_module("benchmark.evaluation.profile")
        model = _FourInputModel().eval()
        loader = [
            (
                torch.randn(2, 4, 3),
                torch.randn(2, 2, 3),
                torch.randn(2, 4, 2),
                torch.randn(2, 2, 2),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "profile.txt"
            with (
                patch.object(profile, "_try_torchinfo_summary", return_value="summary"),
                patch.object(profile, "_try_flops", return_value="flops"),
                patch.object(profile, "_latency_benchmark", return_value=["latency"]),
            ):
                profile.profile_model(
                    model, loader, torch.device("cpu"), 0, 2, str(target)
                )
            self.assertTrue(target.is_file())
            self.assertIn("summary", target.read_text(encoding="utf-8"))
        self.assertFalse(model.training)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 4)

    def test_declared_training_objective_replaces_configured_criterion(self) -> None:
        spec = MODEL_CATALOG.get("DistDF")
        model = spec.model_class(8, 3, 2)
        batch_x = torch.randn(4, 8, 2)
        batch_y = torch.randn(4, 3, 2)

        def forbidden_criterion(*_args):
            raise AssertionError("configured criterion must not replace paper objective")

        outputs, loss = _forward_training(
            model,
            spec.training_objective,
            batch_x,
            None,
            torch.zeros_like(batch_y),
            None,
            batch_y,
            3,
            "M",
            forbidden_criterion,
        )
        self.assertEqual(tuple(outputs.shape), tuple(batch_y.shape))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
