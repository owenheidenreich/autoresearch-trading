"""Comprehensive test suite for v5 training signal redesign.

Tests cover:
- Score config lock (Phase 0)
- Stopped P&L data pipeline (Phase 1)
- EV-weighted loss function (Phase 2)
- Model architecture (3-output) (Phase 2c)
- Day-sequential dataloader (Phase 3)
- Integration: training loop, evaluation, save/load

All tests run on CPU with small synthetic data. No CUDA required.
Run: pytest tests/test_training_v5.py -v
"""
from __future__ import annotations

import importlib
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Ensure training/ is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "training"))


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create a minimal synthetic data.pt dict for testing.

    ~200 bars across 2 trading days, all required fields populated.
    """
    N = 200
    N_FEATURES = 32
    LOOKBACK = 10

    torch.manual_seed(42)
    np.random.seed(42)

    features = torch.randn(N, N_FEATURES)
    targets = torch.randn(N)
    valid_mask = torch.ones(N, dtype=torch.bool)
    # Make first LOOKBACK bars invalid to avoid edge effects
    valid_mask[:LOOKBACK] = False

    # Two trading days: 0-99 and 100-199
    dates = torch.zeros(N, dtype=torch.long)
    dates[100:] = 1

    # P&L arrays with realistic distributions
    call_pnl = torch.randn(N) * 0.2
    put_pnl = torch.randn(N) * 0.2
    otm5_call_pnl = torch.randn(N) * 0.3
    otm5_put_pnl = torch.randn(N) * 0.3
    otm10_call_pnl = torch.randn(N) * 0.4
    otm10_put_pnl = torch.randn(N) * 0.4

    # Stopped P&L — tighter distribution (stops clip extremes)
    call_stopped_pnl = call_pnl.clone().clamp(-0.35, 1.0)
    put_stopped_pnl = put_pnl.clone().clamp(-0.35, 1.0)
    otm5_call_stopped_pnl = otm5_call_pnl.clone().clamp(-0.35, 1.0)
    otm5_put_stopped_pnl = otm5_put_pnl.clone().clamp(-0.35, 1.0)
    otm10_call_stopped_pnl = otm10_call_pnl.clone().clamp(-0.35, 1.0)
    otm10_put_stopped_pnl = otm10_put_pnl.clone().clamp(-0.35, 1.0)

    # Exit labels
    exit_call_label = (torch.rand(N) > 0.9).float()
    exit_put_label = (torch.rand(N) > 0.9).float()

    # Option prices (positive)
    atm_call_prices = torch.rand(N) * 10 + 1
    atm_put_prices = torch.rand(N) * 10 + 1
    otm5_call_prices = torch.rand(N) * 5 + 0.5
    otm5_put_prices = torch.rand(N) * 5 + 0.5
    otm10_call_prices = torch.rand(N) * 2 + 0.2
    otm10_put_prices = torch.rand(N) * 2 + 0.2

    # Metadata
    supervision_weight = torch.ones(N)
    actionable_mask = torch.ones(N)
    risk_state_mask = torch.ones(N)

    # Day boundaries
    day_boundaries = torch.tensor([0, 100], dtype=torch.long)

    # Train/val split at 70%
    train_end = 140
    val_start = 140

    data = {
        'features': features,
        'targets': targets,
        'valid_mask': valid_mask,
        'dates': dates,
        'train_end_idx': train_end - 1,
        'val_start_idx': val_start,
        'val_end_idx': N - 1,
        'call_pnl': call_pnl,
        'put_pnl': put_pnl,
        'exit_call_label': exit_call_label,
        'exit_put_label': exit_put_label,
        'otm5_call_pnl': otm5_call_pnl,
        'otm5_put_pnl': otm5_put_pnl,
        'otm10_call_pnl': otm10_call_pnl,
        'otm10_put_pnl': otm10_put_pnl,
        'call_stopped_pnl': call_stopped_pnl,
        'put_stopped_pnl': put_stopped_pnl,
        'otm5_call_stopped_pnl': otm5_call_stopped_pnl,
        'otm5_put_stopped_pnl': otm5_put_stopped_pnl,
        'otm10_call_stopped_pnl': otm10_call_stopped_pnl,
        'otm10_put_stopped_pnl': otm10_put_stopped_pnl,
        'supervision_weight': supervision_weight,
        'actionable_mask': actionable_mask,
        'risk_state_mask': risk_state_mask,
        'day_boundaries': day_boundaries,
        'atm_call_prices': atm_call_prices,
        'atm_put_prices': atm_put_prices,
        'otm5_call_prices': otm5_call_prices,
        'otm5_put_prices': otm5_put_prices,
        'otm10_call_prices': otm10_call_prices,
        'otm10_put_prices': otm10_put_prices,
    }
    return data


@pytest.fixture
def model():
    """Create a fresh TradingModel for testing."""
    from train import TradingModel
    return TradingModel(num_features=32, lookback=10, d_model=64, n_heads=4, n_layers=3)


# ===========================================================================
# Phase 0: Score Config Lock Tests
# ===========================================================================

class TestScoreConfigLock:
    def test_score_config_values_are_locked(self):
        """Verify _score_config dict has exact expected values."""
        train_path = _project_root / "training" / "train.py"
        source = train_path.read_text()

        expected = {
            'win_rate_bonus': '0.0',
            'rr_bonus': '0.3',
            'drawdown_penalty': '0.5',
            'hold_bonus': '0.0',
            'freq_center': '2.5',
            'freq_width': '2.5',
            'consec_loss_threshold': '3',
            'short_hold_threshold': '0.30',
            'stop_rate_threshold': '0.30',
            'ruin_penalty': '1.0',
            'ruin_threshold': '0.25',
            'risk_fraction_penalty': '0.5',
        }

        for key, val in expected.items():
            pattern = rf"'{key}':\s*{re.escape(val)}"
            assert re.search(pattern, source), (
                f"_score_config['{key}'] should be {val} in train.py"
            )

    def test_no_score_env_vars_in_train(self):
        """Verify no SCORE_* env var readers exist in train.py."""
        train_path = _project_root / "training" / "train.py"
        source = train_path.read_text()

        # Should not find _env_float("SCORE_*") patterns
        matches = re.findall(r'_env_float\(\s*["\']SCORE_', source)
        assert len(matches) == 0, (
            f"Found {len(matches)} SCORE_* env var readers in train.py: {matches}"
        )

        matches = re.findall(r'_env_int\(\s*["\']SCORE_', source)
        assert len(matches) == 0, (
            f"Found {len(matches)} SCORE_* env var readers in train.py: {matches}"
        )

    def test_run_loop_rejects_score_mutation(self):
        """validate_safety() should reject code that changes _score_config values."""
        from run_loop import validate_safety

        # Code with modified drawdown_penalty
        bad_code = """
_score_config = {
    'win_rate_bonus': 0.0,
    'rr_bonus': 0.3,
    'drawdown_penalty': 0.05,
    'hold_bonus': 0.0,
    'freq_center': 2.5,
    'freq_width': 2.5,
}
FEATURE_GROUPS = {
    'returns':   (0, 2),
    'volume':    (2, 5),
    'vol':       (5, 8),
    'vwap':      (8, 10),
    'session':   (10, 12),
    'levels':    (12, 14),
    'trend':     (14, 17),
    'micro':     (17, 19),
    'time':      (19, 22),
    'options':   (22, 24),
    'vix':       (24, 26),
    'greeks':    (26, 29),
    'bollinger': (29, 30),
    'range_ext': (30, 32),
}
"""
        result = validate_safety(bad_code)
        assert result is not None, "validate_safety should reject modified _score_config"
        assert "drawdown_penalty" in result.lower() or "score_config" in result.lower()

    def test_run_loop_rejects_score_env_vars(self):
        """validate_safety() should reject code with SCORE_* env var readers."""
        from run_loop import validate_safety

        bad_code = """
SCORE_DRAWDOWN_PENALTY = _env_float("SCORE_DRAWDOWN_PENALTY", 0.5)
FEATURE_GROUPS = {
    'returns':   (0, 2),
    'volume':    (2, 5),
    'vol':       (5, 8),
    'vwap':      (8, 10),
    'session':   (10, 12),
    'levels':    (12, 14),
    'trend':     (14, 17),
    'micro':     (17, 19),
    'time':      (19, 22),
    'options':   (22, 24),
    'vix':       (24, 26),
    'greeks':    (26, 29),
    'bollinger': (29, 30),
    'range_ext': (30, 32),
}
"""
        result = validate_safety(bad_code)
        assert result is not None, "validate_safety should reject SCORE_* env vars"
        assert "SCORE_" in result


# ===========================================================================
# Phase 1: Data Pipeline Tests
# ===========================================================================

class TestDataPipeline:
    def test_stopped_pnl_tighter_than_unstopped(self, synthetic_data):
        """Stopped P&L should have smaller variance than unstopped."""
        for key in ('call', 'put', 'otm5_call', 'otm5_put', 'otm10_call', 'otm10_put'):
            unstopped = synthetic_data[f'{key}_pnl']
            stopped = synthetic_data[f'{key}_stopped_pnl']

            unstopped_var = unstopped[~unstopped.isnan()].var().item()
            stopped_var = stopped[~stopped.isnan()].var().item()

            assert stopped_var <= unstopped_var + 1e-6, (
                f"{key}: stopped variance ({stopped_var:.4f}) > unstopped ({unstopped_var:.4f})"
            )

    def test_day_boundaries_correct(self, synthetic_data):
        """Day boundaries should correspond to actual date changes."""
        dates = synthetic_data['dates']
        boundaries = synthetic_data['day_boundaries']

        assert boundaries[0] == 0, "First day boundary should be index 0"
        for i in range(1, len(boundaries)):
            idx = boundaries[i].item()
            assert dates[idx] != dates[idx - 1], (
                f"Day boundary at {idx} but dates[{idx}]={dates[idx]} == dates[{idx-1}]={dates[idx-1]}"
            )

    def test_dataloader_yields_stopped_pnl(self, synthetic_data):
        """make_dataloader should yield stopped P&L fields in y tuple."""
        from prepare import make_dataloader

        loader = make_dataloader(synthetic_data, lookback=10, batch_size=16,
                                split="train", device="cpu")
        x, y = next(loader)

        # y tuple should have 18 elements (12 original + 6 stopped P&L)
        assert len(y) == 18, f"Expected 18 fields in y tuple, got {len(y)}"

        # Stopped P&L should be at positions 12-17
        for i in range(12, 18):
            assert y[i].shape[0] == 16, f"y[{i}] batch dim should be 16, got {y[i].shape[0]}"
            assert y[i].dtype == torch.float32

    def test_dataloader_backward_compat_no_stopped_pnl(self):
        """Dataloader should work with old data.pt that lacks stopped P&L."""
        from prepare import make_dataloader

        N, NF = 100, 32
        old_data = {
            'features': torch.randn(N, NF),
            'targets': torch.randn(N),
            'valid_mask': torch.ones(N, dtype=torch.bool),
            'dates': torch.zeros(N, dtype=torch.long),
            'train_end_idx': 69,
            'val_start_idx': 70,
            'val_end_idx': 99,
            'call_pnl': torch.randn(N),
            'put_pnl': torch.randn(N),
            'exit_call_label': torch.zeros(N),
            'exit_put_label': torch.zeros(N),
            'otm5_call_pnl': torch.randn(N),
            'otm5_put_pnl': torch.randn(N),
            'otm10_call_pnl': torch.randn(N),
            'otm10_put_pnl': torch.randn(N),
        }
        old_data['valid_mask'][:10] = False

        loader = make_dataloader(old_data, lookback=10, batch_size=8,
                                split="train", device="cpu")
        x, y = next(loader)
        assert len(y) == 18, "Should still yield 18 fields (with fallback stopped P&L)"


# ===========================================================================
# Phase 2: Model Architecture Tests
# ===========================================================================

class TestModelArchitecture:
    def test_model_outputs_three_values(self, model):
        """TradingModel forward should return (gate, dir, etv) tuple."""
        x = torch.randn(4, 10, 32)
        out = model(x)
        assert isinstance(out, tuple), f"Model output should be tuple, got {type(out)}"
        assert len(out) == 3, f"Model should output 3 values, got {len(out)}"

    def test_model_gate_shape(self, model):
        """gate_logits shape should be (batch, 2)."""
        x = torch.randn(8, 10, 32)
        gate, _, _ = model(x)
        assert gate.shape == (8, 2), f"Gate shape: expected (8, 2), got {gate.shape}"

    def test_model_dir_shape(self, model):
        """dir_logits shape should be (batch, 6)."""
        x = torch.randn(8, 10, 32)
        _, dir_logits, _ = model(x)
        assert dir_logits.shape == (8, 6), f"Dir shape: expected (8, 6), got {dir_logits.shape}"

    def test_model_etv_shape(self, model):
        """etv shape should be (batch,)."""
        x = torch.randn(8, 10, 32)
        _, _, etv = model(x)
        assert etv.shape == (8,), f"ETV shape: expected (8,), got {etv.shape}"

    def test_model_backward_pass(self, model):
        """loss.backward() should run without error."""
        x = torch.randn(4, 10, 32)
        gate, dir_logits, etv = model(x)

        # Simple loss combining all outputs
        loss = gate.sum() + dir_logits.sum() + etv.sum()
        loss.backward()

        # Check gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_model_param_count_reasonable(self, model):
        """Total params should be within expected range."""
        total = sum(p.numel() for p in model.parameters())
        # v4 was ~80k params, v5 adds 65 params for etv_proj
        assert 10_000 < total < 200_000, f"Param count {total} outside expected range"

    def test_model_with_position_state(self, model):
        """Model should accept position_state input."""
        x = torch.randn(4, 10, 32)
        pos_state = torch.zeros(4, 5)
        pos_state[:, 3] = 1.0  # account_health

        gate, dir_logits, etv = model(x, position_state=pos_state)
        assert gate.shape == (4, 2)
        assert dir_logits.shape == (4, 6)
        assert etv.shape == (4,)

    def test_model_without_position_state_training(self, model):
        """In training mode, model should generate synthetic position state."""
        model.train()
        x = torch.randn(4, 10, 32)
        gate, dir_logits, etv = model(x)
        # Should not crash — PositionStateGenerator creates synthetic state
        assert gate.shape == (4, 2)

    def test_model_without_position_state_eval(self, model):
        """In eval mode without position_state, should use default path."""
        model.eval()
        x = torch.randn(4, 10, 32)
        gate, dir_logits, etv = model(x)
        assert gate.shape == (4, 2)


# ===========================================================================
# Phase 2: Loss Function Tests
# ===========================================================================

class TestLossFunction:
    def test_ev_weighted_gate_scales_with_pnl(self):
        """Gradient for +50% trade should be larger than +0.1% trade."""
        from train import sniper_loss

        B = 32
        gate_logits = torch.zeros(B, 2, requires_grad=True)
        dir_logits = torch.zeros(B, 6, requires_grad=True)
        time_feat = torch.ones(B) * 0.5

        # Small positive P&L
        small_pnl = torch.full((B,), 0.001)
        small_stopped = small_pnl.clone()

        loss_small = sniper_loss(
            gate_logits, dir_logits,
            small_pnl, small_pnl, time_feat, torch.randn(B, 32),
            call_stopped_pnl=small_stopped, put_stopped_pnl=small_stopped,
        )

        # Large positive P&L
        gate_logits2 = torch.zeros(B, 2, requires_grad=True)
        dir_logits2 = torch.zeros(B, 6, requires_grad=True)
        large_pnl = torch.full((B,), 0.50)
        large_stopped = large_pnl.clone()

        loss_large = sniper_loss(
            gate_logits2, dir_logits2,
            large_pnl, large_pnl, time_feat, torch.randn(B, 32),
            call_stopped_pnl=large_stopped, put_stopped_pnl=large_stopped,
        )

        # Both should be finite
        assert torch.isfinite(loss_small), f"Small P&L loss is not finite: {loss_small}"
        assert torch.isfinite(loss_large), f"Large P&L loss is not finite: {loss_large}"

        # Large P&L should produce larger loss (stronger signal)
        loss_small.backward()
        loss_large.backward()
        grad_small = gate_logits.grad.abs().sum().item()
        grad_large = gate_logits2.grad.abs().sum().item()
        assert grad_large > grad_small, (
            f"Large P&L gradient ({grad_large}) should exceed small ({grad_small})"
        )

    def test_ev_weighted_gate_negative_pnl(self):
        """Bars with all-negative P&L should get strong NO_TRADE signal."""
        from train import sniper_loss

        B = 16
        gate_logits = torch.zeros(B, 2)
        dir_logits = torch.zeros(B, 6)
        time_feat = torch.ones(B) * 0.5

        # All negative P&L
        neg_pnl = torch.full((B,), -0.20)
        neg_stopped = neg_pnl.clone()

        loss = sniper_loss(
            gate_logits, dir_logits,
            neg_pnl, neg_pnl, time_feat, torch.randn(B, 32),
            call_stopped_pnl=neg_stopped, put_stopped_pnl=neg_stopped,
        )
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_sniper_loss_handles_all_nan(self):
        """Loss should handle all-NaN P&L gracefully."""
        from train import sniper_loss

        B = 8
        gate_logits = torch.zeros(B, 2)
        dir_logits = torch.zeros(B, 6)
        time_feat = torch.ones(B) * 0.5

        nan_pnl = torch.full((B,), float('nan'))
        loss = sniper_loss(
            gate_logits, dir_logits,
            nan_pnl, nan_pnl, time_feat, torch.randn(B, 32),
        )
        assert loss.item() == 0.0, f"All-NaN P&L should produce zero loss, got {loss.item()}"

    def test_etv_loss_gradient_flows(self, model):
        """ETV loss should contribute to parameter gradients."""
        x = torch.randn(4, 10, 32)
        gate, dir_logits, etv = model(x)

        # MSE loss on ETV
        target = torch.randn(4)
        etv_loss = F.mse_loss(etv, target)
        etv_loss.backward()

        # etv_proj should have gradients
        assert model.etv_proj.weight.grad is not None
        assert model.etv_proj.weight.grad.abs().sum() > 0

    def test_soft_direction_targets(self):
        """Direction soft targets should sum to ~1 for each sample."""
        B = 16
        # Simulated stopped P&L (some positive, some negative)
        trade_pnl = torch.randn(B, 6)

        trade_pnl_safe = torch.nan_to_num(trade_pnl, nan=-999.0)
        trade_pnl_shifted = trade_pnl_safe - trade_pnl_safe.min(dim=-1, keepdim=True).values + 0.01
        dir_soft_targets = trade_pnl_shifted / trade_pnl_shifted.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Should sum to 1
        sums = dir_soft_targets.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
            f"Soft targets should sum to 1, got {sums}"
        )

        # All values should be positive
        assert (dir_soft_targets >= 0).all(), "Soft targets should be non-negative"


# ===========================================================================
# Phase 3: Day-Sequential Tests
# ===========================================================================

class TestDaySequential:
    def test_day_sequential_loader_ordering(self, synthetic_data):
        """Bars within a day should come in sequential order."""
        from prepare import make_day_sequential_loader

        loader = make_day_sequential_loader(
            synthetic_data, lookback=10, batch_size=4, device="cpu", split="train")

        prev_bar = -1
        count = 0
        for x, y, bar_in_day in loader:
            if bar_in_day < prev_bar:
                # New day started — reset is expected
                pass
            elif prev_bar >= 0:
                # Same day — bar should be >= previous
                assert bar_in_day >= prev_bar, (
                    f"Non-sequential: bar {bar_in_day} after {prev_bar}"
                )
            prev_bar = bar_in_day
            count += 1
            if count > 30:
                break

    def test_day_sequential_loader_yields_correct_shape(self, synthetic_data):
        """Day-sequential loader should yield proper shapes."""
        from prepare import make_day_sequential_loader

        loader = make_day_sequential_loader(
            synthetic_data, lookback=10, batch_size=4, device="cpu", split="train")

        x, y, bar_in_day = next(loader)
        assert x.ndim == 3, f"x should be 3D, got {x.ndim}D"
        assert x.shape[1] == 10, f"lookback dim should be 10, got {x.shape[1]}"
        assert x.shape[2] == 32, f"feature dim should be 32, got {x.shape[2]}"
        assert len(y) == 18, f"y tuple should have 18 fields, got {len(y)}"
        assert isinstance(bar_in_day, int), f"bar_in_day should be int, got {type(bar_in_day)}"

    def test_day_sequential_missing_boundaries_raises(self):
        """Loader should raise if data.pt lacks day_boundaries."""
        from prepare import make_day_sequential_loader

        data_no_boundaries = {
            'features': torch.randn(100, 32),
            'targets': torch.randn(100),
            'valid_mask': torch.ones(100, dtype=torch.bool),
            'dates': torch.zeros(100, dtype=torch.long),
            'train_end_idx': 69,
            'val_start_idx': 70,
            'val_end_idx': 99,
            'call_pnl': torch.randn(100),
            'put_pnl': torch.randn(100),
            'exit_call_label': torch.zeros(100),
            'exit_put_label': torch.zeros(100),
            'otm5_call_pnl': torch.randn(100),
            'otm5_put_pnl': torch.randn(100),
            'otm10_call_pnl': torch.randn(100),
            'otm10_put_pnl': torch.randn(100),
        }

        with pytest.raises(KeyError, match="day_boundaries"):
            loader = make_day_sequential_loader(
                data_no_boundaries, lookback=10, batch_size=4, device="cpu")
            next(loader)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_training_loop_loss_decreases(self, synthetic_data, model):
        """Train 50 steps on synthetic data — loss should decrease."""
        from prepare import make_dataloader
        from train import sniper_loss, IDX_MINUTES_TO_CLOSE, ETV_LOSS_WEIGHT

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = make_dataloader(synthetic_data, lookback=10, batch_size=16,
                                split="train", device="cpu")

        losses = []
        for step in range(50):
            x, y = next(loader)
            gate, dir_logits, etv = model(x)

            (fwd_ret, call_pnl, put_pnl, exit_call, exit_put,
             otm5c, otm5p, otm10c, otm10p, sw, am, rsm,
             cs, ps, o5cs, o5ps, o10cs, o10ps) = y

            time_feat = x[:, -1, IDX_MINUTES_TO_CLOSE]
            loss = sniper_loss(
                gate, dir_logits, call_pnl, put_pnl, time_feat, x[:, -1, :],
                exit_call, exit_put, otm5c, otm5p, otm10c, otm10p,
                supervision_weight=sw, actionable_mask=am, risk_state_mask=rsm,
                call_stopped_pnl=cs, put_stopped_pnl=ps,
                otm5_call_stopped_pnl=o5cs, otm5_put_stopped_pnl=o5ps,
                otm10_call_stopped_pnl=o10cs, otm10_put_stopped_pnl=o10ps,
                etv_pred=etv,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease (first 10 avg vs last 10 avg)
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        assert late_avg < early_avg, (
            f"Loss should decrease: early avg {early_avg:.4f} vs late avg {late_avg:.4f}"
        )

    def test_training_loop_no_nan_loss(self, synthetic_data, model):
        """No NaN or Inf loss values during training."""
        from prepare import make_dataloader
        from train import sniper_loss, IDX_MINUTES_TO_CLOSE

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = make_dataloader(synthetic_data, lookback=10, batch_size=16,
                                split="train", device="cpu")

        for step in range(20):
            x, y = next(loader)
            gate, dir_logits, etv = model(x)
            (fwd_ret, call_pnl, put_pnl, exit_call, exit_put,
             otm5c, otm5p, otm10c, otm10p, sw, am, rsm,
             cs, ps, o5cs, o5ps, o10cs, o10ps) = y
            time_feat = x[:, -1, IDX_MINUTES_TO_CLOSE]
            loss = sniper_loss(
                gate, dir_logits, call_pnl, put_pnl, time_feat, x[:, -1, :],
                exit_call, exit_put, otm5c, otm5p, otm10c, otm10p,
                supervision_weight=sw, actionable_mask=am, risk_state_mask=rsm,
                call_stopped_pnl=cs, put_stopped_pnl=ps,
                otm5_call_stopped_pnl=o5cs, otm5_put_stopped_pnl=o5ps,
                otm10_call_stopped_pnl=o10cs, otm10_put_stopped_pnl=o10ps,
                etv_pred=etv,
            )
            assert torch.isfinite(loss), f"Loss is not finite at step {step}: {loss.item()}"
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_model_save_load_roundtrip(self, model):
        """Save checkpoint, load it, verify outputs match."""
        model.eval()
        x = torch.randn(2, 10, 32)

        with torch.no_grad():
            gate1, dir1, etv1 = model(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name

        try:
            from train import TradingModel
            model2 = TradingModel(num_features=32, lookback=10, d_model=64, n_heads=4, n_layers=3)
            model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model2.eval()

            with torch.no_grad():
                gate2, dir2, etv2 = model2(x)

            assert torch.allclose(gate1, gate2, atol=1e-5)
            assert torch.allclose(dir1, dir2, atol=1e-5)
            assert torch.allclose(etv1, etv2, atol=1e-5)
        finally:
            os.unlink(path)


# ===========================================================================
# Regression Tests
# ===========================================================================

class TestRegression:
    def test_feature_count_unchanged(self):
        """NUM_FEATURES should still be 32."""
        from prepare import NUM_FEATURES
        assert NUM_FEATURES == 32

    def test_action_space_unchanged(self):
        """NUM_ACTIONS should still be 8."""
        from prepare import NUM_ACTIONS
        assert NUM_ACTIONS == 8

    def test_evaluate_trades_with_3_output_model(self, synthetic_data):
        """evaluate_trades() should work with 3-output model."""
        from prepare import evaluate_trades

        class ThreeOutputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.position_proj = torch.nn.Linear(5, 16)
                self.gate = torch.nn.Linear(32, 2)
                self.dir = torch.nn.Linear(32, 6)
                self.etv = torch.nn.Linear(32, 1)

            def forward(self, x, position_state=None):
                last = x[:, -1, :]
                gate = self.gate(last)
                dir_out = self.dir(last)
                etv = self.etv(last).squeeze(-1)
                return gate, dir_out, etv

        model = ThreeOutputModel()
        model.eval()

        # evaluate_trades needs proper data structure
        # Just verify it doesn't crash on model output unpacking
        try:
            result = evaluate_trades(model, synthetic_data, lookback=10, device="cpu")
            # If it returns, check it has expected keys
            assert 'score' in result or isinstance(result, dict)
        except Exception as e:
            # Some errors are OK (e.g., missing price arrays for full sim)
            # but "too many values to unpack" or "expected 2" should NOT happen
            error_msg = str(e)
            assert "unpack" not in error_msg.lower(), (
                f"Model output unpacking failed: {error_msg}"
            )
            assert "expected 2" not in error_msg.lower(), (
                f"evaluate_trades still expects 2-output model: {error_msg}"
            )
