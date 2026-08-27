"""Contracts for the STID/MoFo pinned-source parity harness."""

from __future__ import annotations

import unittest

import torch

from models.mofo.model import Model as MoFo
from models.stid.model import Model as STID


class SpecialUpstreamParityContracts(unittest.TestCase):
    def test_stid_wrapper_uses_all_three_upstream_input_channels(self):
        model = STID(12, 6, 4, input_dim=3, embed_dim=8, num_layers=1)
        self.assertEqual(model.net.time_series_emb_layer.in_channels, 36)
        self.assertEqual(model.net.encoder[0].drop.p, 0.15)
        values = torch.randn(2, 12, 4, requires_grad=True)
        marks = torch.zeros(2, 12, 6)
        marks[..., 3] = 2
        marks[..., 4] = 5
        output = model(values, marks)
        self.assertEqual(output.shape, (2, 6, 4))
        output.sum().backward()
        self.assertIsNotNone(values.grad)

    def test_mofo_raw_mark_adapter_recovers_periodic_position(self):
        model = MoFo(48, 24, 3, d_model=8, periodic=24, head=2)
        marks = torch.zeros(2, 48, 6)
        marks[..., 3] = 4
        marks[..., 4] = 17
        synthetic = model._build_marks(marks)
        recovered = torch.round((synthetic[:, -1, 0] + 0.5) * 23)
        self.assertTrue(torch.equal(recovered, torch.full((2,), 17.0)))


if __name__ == "__main__":
    unittest.main()
