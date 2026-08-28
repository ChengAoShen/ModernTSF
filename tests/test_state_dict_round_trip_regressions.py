"""Regression coverage for model state that used to escape ``state_dict``."""

from __future__ import annotations

import io
import unittest

import numpy as np
import torch

from benchmark.model_contracts import audit_model_contracts
from models.fedformer.model import FrequencyEnhancedAttention, FrequencyEnhancedBlock
from models.koopa.model import Model as Koopa
from models.pcdcnet._upstream import GCNLayer


class StateDictRoundTripRegressionTests(unittest.TestCase):
    def test_strict_round_trip_for_affected_catalog_models(self) -> None:
        failures = audit_model_contracts(
            names=["DCRNN", "GTS", "FEDformer", "Koopa", "PCDCNet"],
            strict=True,
        )
        self.assertEqual(failures, [])

    def test_fourier_mode_indices_are_persistent_buffers(self) -> None:
        source_block = FrequencyEnhancedBlock(8, 2, 96, 8, "random")
        source_cross = FrequencyEnhancedAttention(8, 2, 60, 96, 8, "random")

        block_state = source_block.state_dict()
        cross_state = source_cross.state_dict()
        self.assertIn("mode_indices", block_state)
        self.assertIn("query_modes", cross_state)
        self.assertIn("key_modes", cross_state)

        restored_block = FrequencyEnhancedBlock(8, 2, 96, 8, "random")
        restored_cross = FrequencyEnhancedAttention(8, 2, 60, 96, 8, "random")
        restored_block.load_state_dict(block_state, strict=True)
        restored_cross.load_state_dict(cross_state, strict=True)
        torch.testing.assert_close(restored_block.mode_indices, source_block.mode_indices)
        torch.testing.assert_close(restored_cross.query_modes, source_cross.query_modes)
        torch.testing.assert_close(restored_cross.key_modes, source_cross.key_modes)

    def test_koopa_first_batch_mask_survives_serialization(self) -> None:
        torch.manual_seed(3)
        source = Koopa(
            seq_len=24,
            pred_len=6,
            enc_in=2,
            dynamic_dim=8,
            hidden_dim=8,
            num_blocks=1,
        ).eval()
        first_batch = torch.randn(2, 24, 2)
        source(first_batch)

        payload = io.BytesIO()
        torch.save(source.state_dict(), payload)
        payload.seek(0)
        restored = Koopa(
            seq_len=24,
            pred_len=6,
            enc_in=2,
            dynamic_dim=8,
            hidden_dim=8,
            num_blocks=1,
        ).eval()
        restored.load_state_dict(
            torch.load(payload, weights_only=True), strict=True
        )

        self.assertTrue(restored.mask_spectrum_initialized.item())
        torch.testing.assert_close(restored.mask_spectrum, source.mask_spectrum)
        comparison_batch = torch.randn(1, 24, 2)
        torch.testing.assert_close(restored(comparison_batch), source(comparison_batch))

    def test_pcdc_graph_convolution_eval_does_not_drop_or_cache_edges(self) -> None:
        layer = GCNLayer(
            in_features=4,
            out_features=4,
            gso=torch.eye(3),
            num_layers=1,
            dropout=0.0,
            drop_edge_p=0.9,
        ).eval()
        values = torch.randn(2, 3, 4)
        torch.manual_seed(4)
        first = layer(values)
        torch.manual_seed(5)
        second = layer(values)
        torch.testing.assert_close(second, first, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
