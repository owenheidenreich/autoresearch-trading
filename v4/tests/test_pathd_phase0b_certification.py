from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v4.research.pathd_causal_account_state import (
    AccountEvent,
    CausalAccountState,
    account_feature_snapshot,
    apply_account_event,
    replay_account_events,
)
from v4.research.pathd_feature_admission_ledger import (
    ADMITTED,
    AdmissionLedgerError,
    DEFAULT_LEDGER_PATH,
    admitted_feature_matrix,
    verify_ledger,
)
from v4.research.pathd_opra_parity_features import (
    bsm_price,
    historical_parity_snapshot,
    historical_self_computed_greeks,
    implied_spot_feature_frame,
    iv_snapshot,
    live_parity_snapshot,
    live_self_computed_greeks,
    solve_implied_volatility,
)
from v4.research.pathd_phase0b_certification import REPORT_PATH


def _synthetic_ladder(decision_time: datetime, *, spot: float = 100.0, sigma: float = 0.25) -> pd.DataFrame:
    years = (datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc) - decision_time).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
    rows = []
    for strike in range(80, 125, 5):
        for right in ("C", "P"):
            mid = bsm_price(
                spot=spot,
                strike=float(strike),
                years=years,
                rate=0.04,
                dividend=0.0,
                volatility=sigma,
                right=right,
            )
            half_spread = min(0.005, mid / 2.0)
            rows.append(
                {
                    "event_time": pd.Timestamp(decision_time),
                    "raw_symbol": f"SYNTH-{strike}-{right}",
                    "strike": float(strike),
                    "right": right,
                    "bid": max(0.0, mid - half_spread),
                    "ask": mid + half_spread,
                    "underlying_price": spot,
                }
            )
    return pd.DataFrame(rows)


def test_shared_parity_paths_are_exact_and_ignore_official_spx() -> None:
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    rows = _synthetic_ladder(now)
    historical = historical_parity_snapshot(rows, decision_time=now)
    live = live_parity_snapshot(rows, decision_time=now)
    assert historical == live
    assert historical.implied_spot == pytest.approx(100.0, abs=1e-12)
    mutated = rows.copy()
    mutated["underlying_price"] = 1e12
    assert live_parity_snapshot(mutated, decision_time=now) == live


def test_implied_spot_family_is_causal_under_future_mutation() -> None:
    frames = []
    for minute in range(16):
        now = datetime(2026, 7, 31, 18, minute, tzinfo=timezone.utc)
        frames.append(_synthetic_ladder(now, spot=100.0 + minute * 0.1))
    rows = pd.concat(frames, ignore_index=True)
    baseline = implied_spot_feature_frame(rows)
    mutated = rows.copy()
    cutoff = pd.Timestamp(datetime(2026, 7, 31, 18, 7, tzinfo=timezone.utc))
    future = pd.to_datetime(mutated["event_time"], utc=True) > cutoff
    mutated.loc[future, "bid"] *= 0.5
    mutated.loc[future, "ask"] *= 1.5
    after = implied_spot_feature_frame(mutated)
    columns = [name for name in baseline if name != "event_time"]
    assert np.allclose(
        baseline.loc[baseline["event_time"] <= cutoff, columns],
        after.loc[after["event_time"] <= cutoff, columns],
        equal_nan=True,
        atol=0.0,
        rtol=0.0,
    )
    assert set(columns) == {
        "opra_implied_spot",
        "opra_implied_spot_dispersion_bps",
        "opra_spot_return_1m_bps",
        "opra_spot_return_5m_bps",
        "opra_spot_return_15m_bps",
        "opra_spot_vwap_gap_bps",
        "opra_spot_session_range_bps",
        "opra_spot_omar_clipped",
    }


def test_iv_solver_and_snapshot_recover_frozen_volatility() -> None:
    price = bsm_price(
        spot=100.0, strike=100.0, years=0.25, rate=0.04,
        dividend=0.0, volatility=0.25, right="C",
    )
    assert solve_implied_volatility(
        price=price, spot=100.0, strike=100.0, years=0.25, right="C"
    ) == pytest.approx(0.25, abs=1e-10)
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    snapshot = iv_snapshot(_synthetic_ladder(now), decision_time=now)
    assert snapshot.atm_iv == pytest.approx(0.25, abs=1e-10)


def test_golden_greeks_are_identical_on_both_paths() -> None:
    inputs = {
        "option_price": 7.965567455405804,
        "spot": 100.0,
        "strike": 100.0,
        "years": 1.0,
        "right": "C",
        "rate": 0.0,
        "dividend": 0.0,
    }
    historical = historical_self_computed_greeks(**inputs)
    live = live_self_computed_greeks(**inputs)
    assert historical == live
    assert historical["self_iv"] == pytest.approx(0.2, abs=1e-10)
    assert historical["bs_delta"] == pytest.approx(0.539827837277029, abs=1e-12)
    assert historical["bs_gamma"] == pytest.approx(0.0198476273738506, abs=1e-12)


def test_account_state_serial_and_future_invariance() -> None:
    prefix = (
        AccountEvent(1_000_000_000, "ENTRY_FILL"),
        AccountEvent(2_000_000_000, "EXIT_FILL", -25.0),
    )
    batch = replay_account_events(prefix)
    step = CausalAccountState()
    for event in prefix:
        step = apply_account_event(step, event)
    assert batch == step
    before = account_feature_snapshot(batch, decision_time_ns=2_500_000_000)
    future_a = (*prefix, AccountEvent(3_000_000_000, "ENTRY_FILL"), AccountEvent(4_000_000_000, "EXIT_FILL", 50.0))
    future_b = (*prefix, AccountEvent(3_000_000_000, "ENTRY_FILL"), AccountEvent(4_000_000_000, "EXIT_FILL", -400.0))
    assert replay_account_events(future_a) != replay_account_events(future_b)
    assert account_feature_snapshot(replay_account_events(prefix), decision_time_ns=2_500_000_000) == before
    assert before["remaining_entry_budget_dollars"] == 475.0


def test_phase0b_ledger_parent_order_receipts_and_enforcement() -> None:
    """Corrected 2026-08-04.

    Track-B originally admitted 21 features on receipts that prove the TRANSFORM
    is identical across paths. Those receipts stand. What they do not establish
    is when the transform's INPUTS arrive: implied_spot consumes OPRA bid/ask,
    whose multi-session arrival distribution is uncertified, so the whole chain
    must stay blocked until entry.opra_cbbo1m_native.v1 is admitted.
    """
    payload = verify_ledger(DEFAULT_LEDGER_PATH)
    admitted = [row for row in payload["features"] if row["status"] == ADMITTED]
    assert len(admitted) == 8
    families = {row["contract_id"] for row in admitted}
    assert families == {"entry.contract_clock.v1"}, (
        "no market observable may be admitted while the OPRA arrival clock is uncertified"
    )

    index = {row["name"]: row for row in payload["features"]}
    # Blocked, and blocked for the right reason -- on the parent, not on receipts.
    for name in ("opra_implied_spot", "opra_atm_iv", "bs_delta"):
        assert index[name]["status"] != ADMITTED
        assert "parent_family_not_admitted" in (index[name]["barred_reason"] or "")

    # The completed Track-B transform receipts must be preserved, not discarded.
    assert index["opra_implied_spot"]["receipts"], "Track-B receipts must survive the block"

    # No admitted row may carry a local-compute clock in place of an arrival clock.
    for row in admitted:
        assert row["availability_clock_ms"] is not None
    assert index["entries_so_far"]["status"] == "BARRED"
    assert "historical/live ledger transition identity" in index["entries_so_far"]["barred_reason"]
    assert all(row["availability_clock_ms"] not in {2336, 30603.667} for row in admitted if row["contract_id"] != "entry.contract_clock.v1")
    # Corrected 2026-08-04: the Track-B features are no longer trainable, so the
    # enforcement path must now REFUSE them. This is the regression that matters --
    # a model must not be able to train on a feature whose arrival time is unknown.
    with pytest.raises(AdmissionLedgerError, match=r"opra_implied_spot:BARRED"):
        admitted_feature_matrix(
            pd.DataFrame({"opra_implied_spot": [100.0], "bs_delta": [0.5]}),
            ("opra_implied_spot", "bs_delta"),
        )
    with pytest.raises(AdmissionLedgerError, match=r"option_bid:BARRED"):
        admitted_feature_matrix(pd.DataFrame({"option_bid": [1.0]}), ("option_bid",))
    # The 8 calendar/geometry features remain trainable.
    clock_frame = pd.DataFrame({"minute_of_session": [30], "strike": [5500.0]})
    assert list(
        admitted_feature_matrix(clock_frame, ("minute_of_session", "strike")).columns
    ) == ["minute_of_session", "strike"]
    assert REPORT_PATH.is_file()
    assert "STOP_FOR_CLAUDE_VERIFICATION" in REPORT_PATH.read_text()


def test_existing_paper_round_trip_is_recorded_as_noncertifying() -> None:
    receipt_path = (
        DEFAULT_LEDGER_PATH.parent
        / "phase0b_receipts"
        / "causal_account_state_existing_paper_transition_audit_noncertifying.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "INSUFFICIENT_EXISTING_PAPER_EVIDENCE"
    assert receipt["partial_entry_exit_transition_observed"] is True
    assert receipt["observed_pre_action_position_occupancies"] == [0, 1]
    assert receipt["broker_or_paper_accessed_during_phase0b"] is False
    assert "post-exit account-state snapshot" in receipt["missing_for_six_feature_identity"]
