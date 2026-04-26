"""Mutate-future-bars test for the TCN oracle.

Discipline anchor: 2026-04-25 oracle-gate label-leakage retraction. Every
new model must verify that output[t] depends ONLY on input[≤t]. For a
causal Conv1d stack, the test is mechanical: forward-pass on input X,
randomly mutate input[t+1:], forward-pass again, assert output[t] is
bit-exact identical.

This is the H3c equivalent of the H3a feature look-ahead audit.
"""
from __future__ import annotations

import unittest

import torch

from v3.layer3.tcn_oracle import (
    CausalConv1d,
    TCNBlock,
    TCNOracle,
)


class TestCausalConv1d(unittest.TestCase):
    """Single-layer causality."""

    def test_output_independent_of_future_inputs(self):
        torch.manual_seed(42)
        layer = CausalConv1d(in_channels=3, out_channels=4, kernel_size=3, dilation=1)
        layer.eval()

        x = torch.randn(2, 3, 20)
        with torch.no_grad():
            ref = layer(x)

        for t in range(0, 19):
            x_mut = x.clone()
            # Randomize all positions strictly after t
            x_mut[:, :, t + 1:] = torch.randn(2, 3, 19 - t)
            with torch.no_grad():
                out = layer(x_mut)
            self.assertTrue(
                torch.allclose(out[:, :, t], ref[:, :, t], atol=1e-7),
                f"CausalConv1d violated causality at t={t}",
            )

    def test_dilated_kernel_causality(self):
        """Dilation > 1 must still be causal."""
        torch.manual_seed(42)
        for dilation in [2, 4, 8]:
            layer = CausalConv1d(in_channels=2, out_channels=2, kernel_size=3, dilation=dilation)
            layer.eval()
            x = torch.randn(1, 2, 30)
            with torch.no_grad():
                ref = layer(x)
            for t in range(0, 29):
                x_mut = x.clone()
                x_mut[:, :, t + 1:] = torch.randn(1, 2, 29 - t)
                with torch.no_grad():
                    out = layer(x_mut)
                self.assertTrue(
                    torch.allclose(out[:, :, t], ref[:, :, t], atol=1e-7),
                    f"Dilation {dilation} violated causality at t={t}",
                )

    def test_output_length_equals_input(self):
        for k in [3, 5, 7]:
            for d in [1, 2, 4]:
                layer = CausalConv1d(in_channels=2, out_channels=2, kernel_size=k, dilation=d)
                layer.eval()
                x = torch.randn(1, 2, 50)
                with torch.no_grad():
                    out = layer(x)
                self.assertEqual(out.shape[2], 50, f"k={k}, d={d}: output length {out.shape[2]} != 50")


class TestTCNBlock(unittest.TestCase):
    """Block-level (residual + GELU) causality."""

    def test_residual_block_causality(self):
        torch.manual_seed(42)
        block = TCNBlock(channels=4, kernel_size=3, dilation=2)
        block.eval()
        x = torch.randn(2, 4, 25)
        with torch.no_grad():
            ref = block(x)
        for t in range(0, 24):
            x_mut = x.clone()
            x_mut[:, :, t + 1:] = torch.randn(2, 4, 24 - t)
            with torch.no_grad():
                out = block(x_mut)
            self.assertTrue(
                torch.allclose(out[:, :, t], ref[:, :, t], atol=1e-6),
                f"TCNBlock violated causality at t={t}",
            )


class TestTCNOracle(unittest.TestCase):
    """End-to-end model causality."""

    def test_full_model_causality(self):
        torch.manual_seed(42)
        model = TCNOracle(n_features=10, hidden=8, num_blocks=4, kernel_size=3)
        model.eval()
        x = torch.randn(2, 50, 10)
        with torch.no_grad():
            ref = model(x)

        # Test causality at multiple time points
        for t in [0, 5, 10, 25, 40, 48]:
            x_mut = x.clone()
            x_mut[:, t + 1:, :] = torch.randn(2, 49 - t, 10)
            with torch.no_grad():
                out = model(x_mut)
            self.assertTrue(
                torch.allclose(out[:, t], ref[:, t], atol=1e-6),
                f"TCNOracle violated causality at t={t}, "
                f"max diff={float((out[:, t] - ref[:, t]).abs().max()):.2e}",
            )

    def test_output_length_equals_input_length(self):
        model = TCNOracle(n_features=99, hidden=64, num_blocks=4, kernel_size=3)
        model.eval()
        for T in [10, 50, 100, 200]:
            x = torch.randn(1, T, 99)
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape, (1, T))

    def test_realistic_config_causality(self):
        """The exact config we'll use for the L3 oracle."""
        torch.manual_seed(42)
        model = TCNOracle(n_features=99, hidden=64, num_blocks=4, kernel_size=3)
        model.eval()
        # Realistic trade length range
        x = torch.randn(4, 150, 99)
        with torch.no_grad():
            ref = model(x)
        # Spot-check causality at every 10th bar
        for t in range(0, 149, 10):
            x_mut = x.clone()
            x_mut[:, t + 1:, :] = torch.randn(4, 149 - t, 99)
            with torch.no_grad():
                out = model(x_mut)
            self.assertTrue(
                torch.allclose(out[:, t], ref[:, t], atol=1e-5),
                f"Realistic config violated causality at t={t}",
            )


if __name__ == "__main__":
    unittest.main()
