from __future__ import annotations

import csv
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from v2 import train
from v2.core.chain_data import dynamic_slice_bounds, padded_snapshot_with_slice
from v2.core.decision_trace import build_trace_for_bar, save_traces
from v2.core.policy import DEFAULT_POLICY
from v2.replay import _resolve_gate_threshold, load_model_from_path, model_to_intent


class TestGatePathCleanup(unittest.TestCase):

    def setUp(self):
        self._orig = {
            "GATE_W": train.GATE_W,
            "SEL_W": train.SEL_W,
            "SIDE_SEL_W": train.SIDE_SEL_W,
            "EXACT_W": train.EXACT_W,
            "COMP_W": train.COMP_W,
            "SIDE_W": train.SIDE_W,
            "AGG_W": train.AGG_W,
            "GATE_TARGET_MODE": train.GATE_TARGET_MODE,
            "GATE_PNL_THRESHOLD": train.GATE_PNL_THRESHOLD,
            "GATE_PNL_LOSS_SCALE": train.GATE_PNL_LOSS_SCALE,
            "GATE_POSITIVE_PNL": train.GATE_POSITIVE_PNL,
            "GATE_POSITIVE_MAX_WINNERS": train.GATE_POSITIVE_MAX_WINNERS,
            "BAR_QUALITY_PASS_THRESHOLD": train.BAR_QUALITY_PASS_THRESHOLD,
            "BAR_QUALITY_MIN_PNL": train.BAR_QUALITY_MIN_PNL,
            "BAR_QUALITY_TARGET_PNL": train.BAR_QUALITY_TARGET_PNL,
            "BAR_QUALITY_MFE_TARGET": train.BAR_QUALITY_MFE_TARGET,
            "BAR_QUALITY_MAX_MAE": train.BAR_QUALITY_MAX_MAE,
            "BAR_QUALITY_BREAKEVEN_MAX": train.BAR_QUALITY_BREAKEVEN_MAX,
            "BAR_QUALITY_LOSS_SCALE": train.BAR_QUALITY_LOSS_SCALE,
            "RANK_WEIGHT_BY_BAR_QUALITY": train.RANK_WEIGHT_BY_BAR_QUALITY,
        }
        train.GATE_W = 1.0
        train.SEL_W = 0.0
        train.SIDE_SEL_W = 0.0
        train.EXACT_W = 0.0
        train.COMP_W = 0.0
        train.SIDE_W = 0.0
        train.AGG_W = 0.0

    def tearDown(self):
        for name, value in self._orig.items():
            setattr(train, name, value)

    def _targets(self, labels: torch.Tensor, slice_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, n_contracts = labels.shape
        if slice_mask is None:
            slice_mask = torch.ones(batch, n_contracts, dtype=torch.bool)
        return {
            "contract_labels": labels,
            "contract_slice_mask": slice_mask,
            "best_idx": torch.full((batch,), -1, dtype=torch.long),
            "label_trade": torch.tensor([True, False]),
            "label_trade_valid": torch.tensor([True, True]),
            "contracts_full": torch.zeros(batch, n_contracts, 22),
            "label_quality": torch.ones(batch),
            "label_bar_quality": torch.tensor([0.8, 0.1], dtype=torch.float32),
        }

    def _outputs(self, scores: torch.Tensor, gate_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_raw = scores.clone()
        return {
            "contract_scores": scores,
            "valid_mask": torch.ones_like(scores, dtype=torch.bool),
            "gate_logit": gate_logits,
            "opportunity_logit": gate_logits,
            "side_logit": torch.zeros(scores.size(0)),
            "aggression_logits": torch.zeros(scores.size(0), 3),
            "call_scores_raw": shared_raw,
            "put_scores_raw": shared_raw,
            "is_put": torch.zeros_like(scores, dtype=torch.bool),
        }

    def test_binary_gate_loss_ignores_contract_scores(self):
        train.GATE_TARGET_MODE = "binary"
        labels = torch.zeros(2, 3)
        targets = self._targets(labels)
        gate_logits = torch.tensor([0.7, -0.4])

        loss_a, metrics_a = train.compute_loss(
            self._outputs(torch.tensor([[0.1, 0.2, 0.3], [9.0, -5.0, 1.0]]), gate_logits),
            targets,
        )
        loss_b, metrics_b = train.compute_loss(
            self._outputs(torch.tensor([[7.0, -3.0, 5.0], [-1.0, 2.0, 4.0]]), gate_logits),
            targets,
        )

        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)
        self.assertAlmostEqual(metrics_a["gate"], metrics_b["gate"], places=6)

    def test_max_pnl_gate_loss_ignores_contract_scores(self):
        train.GATE_TARGET_MODE = "max_pnl"
        train.GATE_PNL_THRESHOLD = 0.2
        train.GATE_PNL_LOSS_SCALE = 10.0
        train.GATE_POSITIVE_PNL = 0.30
        train.GATE_POSITIVE_MAX_WINNERS = 2
        labels = torch.tensor([[0.10, 0.35, float("nan")], [0.05, 0.12, 0.08]])
        targets = self._targets(labels)
        gate_logits = torch.tensor([0.1, -0.2])

        loss_a, metrics_a = train.compute_loss(
            self._outputs(torch.tensor([[0.1, 0.2, 0.3], [9.0, -5.0, 1.0]]), gate_logits),
            targets,
        )
        loss_b, metrics_b = train.compute_loss(
            self._outputs(torch.tensor([[7.0, -3.0, 5.0], [-1.0, 2.0, 4.0]]), gate_logits),
            targets,
        )

        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)
        self.assertAlmostEqual(metrics_a["gate"], metrics_b["gate"], places=6)

    def test_max_pnl_gate_ignores_out_of_slice_contracts(self):
        train.GATE_TARGET_MODE = "max_pnl"
        train.GATE_PNL_THRESHOLD = 0.2
        labels = torch.tensor([[0.35, 0.01, float("nan")], [0.05, 0.12, 0.08]])
        slice_mask = torch.tensor([[False, True, False], [True, True, True]])
        targets = self._targets(labels, slice_mask=slice_mask)
        gate_logits = torch.tensor([0.1, -0.2])

        loss, metrics = train.compute_loss(
            self._outputs(torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.1, 0.2]]), gate_logits),
            targets,
        )
        expected = train.GATE_PNL_LOSS_SCALE * torch.nn.functional.mse_loss(
            gate_logits,
            torch.tensor([0.01 - train.GATE_PNL_THRESHOLD, 0.12 - train.GATE_PNL_THRESHOLD]),
        )
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertAlmostEqual(metrics["gate"], expected.item(), places=6)

    def test_bar_quality_gate_loss_ignores_contract_scores(self):
        train.GATE_TARGET_MODE = "bar_quality"
        train.BAR_QUALITY_PASS_THRESHOLD = 0.55
        train.BAR_QUALITY_LOSS_SCALE = 5.0
        labels = torch.tensor([[0.10, 0.35, float("nan")], [0.05, 0.12, 0.08]])
        targets = self._targets(labels)
        gate_logits = torch.tensor([0.2, -0.3])

        loss_a, metrics_a = train.compute_loss(
            self._outputs(torch.tensor([[0.1, 0.2, 0.3], [9.0, -5.0, 1.0]]), gate_logits),
            targets,
        )
        loss_b, metrics_b = train.compute_loss(
            self._outputs(torch.tensor([[7.0, -3.0, 5.0], [-1.0, 2.0, 4.0]]), gate_logits),
            targets,
        )

        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)
        self.assertAlmostEqual(metrics_a["gate"], metrics_b["gate"], places=6)

    def test_model_to_intent_rejects_only_on_live_gate(self):
        intent = model_to_intent(
            gate_logit=torch.tensor(0.0),
            side_logit=torch.tensor(0.0),
            contract_scores=torch.tensor([5.0, 4.0]),
            contract_labels=torch.tensor([0.1, 0.2]),
            valid_mask=torch.tensor([True, True]),
            contract_features=torch.ones(2, 22),
            contract_indices=torch.tensor([10, 11]),
            sidecar={"bar_timestamps": np.array([123]), "date": "2026-01-02"},
            local_bar=0,
            spot_price=500.0,
            policy=DEFAULT_POLICY,
        )
        self.assertFalse(intent.trade)
        self.assertIn("opportunity_reject", intent.reason_codes)

    def test_model_to_intent_ranks_with_effective_inference_scores(self):
        contract_features = torch.ones(2, 22)
        contract_features[0, 14] = 0.0
        contract_features[1, 14] = 1.0
        contract_features[0, 2] = 0.0
        contract_features[1, 2] = 1.0

        with mock.patch("v2.replay.describe_contract") as mock_describe, mock.patch(
            "v2.replay.extract_contract_series"
        ) as mock_series:
            mock_describe.return_value = SimpleNamespace(expiry="2026-01-16", strike=505.0, right="P")
            mock_series.return_value = {
                "mid": np.array([1.25], dtype=np.float32),
                "bid": np.array([1.20], dtype=np.float32),
                "ask": np.array([1.30], dtype=np.float32),
            }
            intent = model_to_intent(
                gate_logit=torch.tensor(0.5),
                side_logit=torch.tensor(0.0),
                contract_scores=torch.tensor([9.0, 1.0]),
                contract_labels=torch.tensor([0.1, 0.2]),
                valid_mask=torch.tensor([True, True]),
                contract_features=contract_features,
                contract_indices=torch.tensor([10, 11]),
                sidecar={"bar_timestamps": np.array([123]), "date": "2026-01-02"},
                local_bar=0,
                spot_price=500.0,
                policy=DEFAULT_POLICY,
            )

        self.assertTrue(intent.trade)
        self.assertEqual(intent.contract_index, 11)
        self.assertEqual(intent.right, "P")

    def test_model_to_intent_respects_slice_mask(self):
        contract_features = torch.ones(2, 22)
        contract_features[:, 14] = 1.0
        with mock.patch("v2.replay.describe_contract") as mock_describe, mock.patch(
            "v2.replay.extract_contract_series"
        ) as mock_series:
            mock_describe.return_value = SimpleNamespace(expiry="2026-01-16", strike=500.0, right="C")
            mock_series.return_value = {
                "mid": np.array([1.25], dtype=np.float32),
                "bid": np.array([1.20], dtype=np.float32),
                "ask": np.array([1.30], dtype=np.float32),
            }
            intent = model_to_intent(
                gate_logit=torch.tensor(0.5),
                side_logit=torch.tensor(0.0),
                contract_scores=torch.tensor([9.0, 1.0]),
                contract_labels=torch.tensor([0.1, 0.2]),
                valid_mask=torch.tensor([True, True]),
                slice_mask=torch.tensor([False, True]),
                contract_features=contract_features,
                contract_indices=torch.tensor([10, 11]),
                sidecar={"bar_timestamps": np.array([123]), "date": "2026-01-02"},
                local_bar=0,
                spot_price=500.0,
                policy=DEFAULT_POLICY,
            )

        self.assertTrue(intent.trade)
        self.assertEqual(intent.contract_index, 11)

    def test_dynamic_slice_contract_helpers(self):
        atm, lo, hi = dynamic_slice_bounds(5107.0)
        self.assertEqual(atm, 5105.0)
        self.assertEqual(lo, 5055.0)
        self.assertEqual(hi, 5155.0)

        sidecar = {
            "bar_ptrs": np.array([0, 3], dtype=np.int32),
            "row_features": np.array(
                [
                    [1.0, 5055.0] + [0.0] * 20,
                    [1.0, 5205.0] + [0.0] * 20,
                    [1.0, 5110.0] + [0.0] * 20,
                ],
                dtype=np.float32,
            ),
            "row_labels": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "row_contract_idx": np.array([0, 1, 2], dtype=np.int32),
            "bar_slice_lo_strike": np.array([5055.0], dtype=np.float32),
            "bar_slice_hi_strike": np.array([5155.0], dtype=np.float32),
        }
        _, _, _, slice_mask = padded_snapshot_with_slice(sidecar, 0, 4)
        self.assertEqual(slice_mask.tolist(), [True, False, True, False])

    def test_sparse_high_conviction_gate_requires_strong_sparse_winner(self):
        sidecar = {
            "bar_ptrs": np.array([0, 4], dtype=np.int32),
            "row_labels": np.array([0.31, -0.05, 0.00, np.nan], dtype=np.float32),
            "row_features": np.array(
                [
                    [1.0, 5000.0] + [0.0] * 20,
                    [1.0, 5005.0] + [0.0] * 20,
                    [1.0, 5010.0] + [0.0] * 20,
                    [0.0, 0.0] + [0.0] * 20,
                ],
                dtype=np.float32,
            ),
            "bar_slice_lo_strike": np.array([4995.0], dtype=np.float32),
            "bar_slice_hi_strike": np.array([5015.0], dtype=np.float32),
        }
        self.assertTrue(train._compute_sparse_high_conviction_opportunity(sidecar, 0))

    def test_sparse_high_conviction_gate_rejects_crowded_positive_slice(self):
        sidecar = {
            "bar_ptrs": np.array([0, 4], dtype=np.int32),
            "row_labels": np.array([0.35, 0.12, 0.07, -0.03], dtype=np.float32),
            "row_features": np.array(
                [
                    [1.0, 5000.0] + [0.0] * 20,
                    [1.0, 5005.0] + [0.0] * 20,
                    [1.0, 5010.0] + [0.0] * 20,
                    [1.0, 5015.0] + [0.0] * 20,
                ],
                dtype=np.float32,
            ),
            "bar_slice_lo_strike": np.array([4995.0], dtype=np.float32),
            "bar_slice_hi_strike": np.array([5020.0], dtype=np.float32),
        }
        self.assertFalse(train._compute_sparse_high_conviction_opportunity(sidecar, 0))

    def test_bar_quality_prefers_clean_sparse_winner(self):
        sidecar = {
            "bar_ptrs": np.array([0, 4], dtype=np.int32),
            "row_labels": np.array([0.42, -0.04, -0.06, np.nan], dtype=np.float32),
            "row_features": np.array(
                [
                    [1.0, 5000.0] + [0.0] * 20,
                    [1.0, 5005.0] + [0.0] * 20,
                    [1.0, 5010.0] + [0.0] * 20,
                    [0.0, 0.0] + [0.0] * 20,
                ],
                dtype=np.float32,
            ),
            "bar_slice_lo_strike": np.array([4995.0], dtype=np.float32),
            "bar_slice_hi_strike": np.array([5015.0], dtype=np.float32),
            "bar_slice_best_contract_idx": np.array([0], dtype=np.int32),
            "row_raw_returns": np.array([[0.10, 0.28], [0.00, -0.02], [0.00, -0.03], [0.0, 0.0]], dtype=np.float32),
            "row_mfe": np.array([[0.08, 0.30], [0.00, 0.01], [0.00, 0.01], [0.0, 0.0]], dtype=np.float32),
            "row_mae": np.array([[-0.02, -0.04], [-0.06, -0.08], [-0.05, -0.09], [0.0, 0.0]], dtype=np.float32),
            "row_bars_to_breakeven": np.array([2.0, 8.0, 10.0, 0.0], dtype=np.float32),
        }
        quality = train._compute_bar_opportunity_quality(sidecar, 0)
        self.assertGreaterEqual(quality, train.BAR_QUALITY_PASS_THRESHOLD)

    def test_linear_score_heads_build_both_linear_heads(self):
        model = train.TradingModel(linear_score_heads=True)
        self.assertIsInstance(model.call_score_head, torch.nn.Linear)
        self.assertIsInstance(model.put_score_head, torch.nn.Linear)

    def test_quantile_gate_threshold_hits_target_pass_rate(self):
        policy = DEFAULT_POLICY.__class__(gate_threshold_mode="quantile", gate_target_pass_rate=0.25)
        thr, pass_rate = _resolve_gate_threshold(policy, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(thr, 2.25, places=6)
        self.assertAlmostEqual(pass_rate, 0.25, places=6)

    def test_quantile_gate_threshold_respects_floor(self):
        policy = DEFAULT_POLICY.__class__(
            gate_threshold_mode="quantile",
            gate_target_pass_rate=0.50,
            gate_threshold_floor=0.0,
        )
        thr, pass_rate = _resolve_gate_threshold(policy, np.array([-3.0, -2.0, -1.0, -0.5], dtype=np.float32))
        self.assertEqual(thr, 0.0)
        self.assertEqual(pass_rate, 0.0)

    def test_trace_schema_contains_live_gate_fields_only(self):
        trace = build_trace_for_bar(
            date="2026-01-02",
            bar_of_day=45,
            global_bar_idx=123,
            spot_price=500.0,
            vix_regime=0.2,
            gate_logit=0.4,
            gate_threshold=0.1,
            contract_scores=np.array([0.5, 0.2], dtype=np.float32),
            valid_mask=np.array([True, True]),
            contract_features=np.array([[1.0] * 22, [1.0] * 22], dtype=np.float32),
            contract_labels=np.array([0.3, 0.1], dtype=np.float32),
            contract_indices=np.array([10, 11], dtype=np.int32),
            decision="trade",
            selected_contract_idx=10,
        )
        self.assertTrue(trace.gate_pass)

        with tempfile.NamedTemporaryFile("w+", suffix=".csv") as tmp:
            save_traces([trace], tmp.name)
            tmp.seek(0)
            header = next(csv.reader(tmp))

        self.assertIn("gate_logit", header)
        self.assertIn("gate_threshold", header)
        self.assertIn("gate_pass", header)
        self.assertNotIn("no_trade_score", header)
        self.assertNotIn("score_delta", header)

    def test_stale_checkpoint_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = train.TradingModel(d_model=64, depth=3, n_heads=4, dropout=0.1)
            state = model.state_dict()
            state["no_trade_head.0.weight"] = torch.zeros(64, 64)
            path = f"{tmp}/stale.pt"
            torch.save(
                {
                    "model_state_dict": state,
                    "hyperparams": {"d_model": 64, "depth": 3, "n_heads": 4, "dropout": 0.1},
                },
                path,
            )

            with self.assertRaisesRegex(RuntimeError, "stale model artifact; retrain required"):
                load_model_from_path(path)


if __name__ == "__main__":
    unittest.main()
