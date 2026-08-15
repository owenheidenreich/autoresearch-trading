from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import causal_day_simulator as sim
from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES


SESSION = "2025-09-03"


def _quotes(*, missing_bid_minutes: set[str] | None = None) -> pd.DataFrame:
    missing_bid_minutes = missing_bid_minutes or set()
    rows = []
    for i, minute in enumerate(QUOTE_MINUTES):
        for right, strike, contract in (
            ("C", 105.0, "call"),
            ("P", 95.0, "put"),
            ("C", 200.0, "deep_call"),
        ):
            bid, ask, mid = (1.0 + i / 100.0, 1.2 + i / 100.0, 1.1 + i / 100.0)
            bid_size = 1.0
            if minute in missing_bid_minutes and contract == "call":
                bid = ask = mid = np.nan
                bid_size = 0.0
            rows.append(
                {
                    "event_time": pd.Timestamp(f"{SESSION} {minute}", tz="America/New_York"),
                    "expiry": SESSION,
                    "contract_id": contract,
                    "raw_symbol": contract,
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "bid_size": bid_size,
                    "ask_size": 1.0,
                    "mid": mid,
                    "quote_age_ms": 0.0,
                    "volume": 1.0,
                    "open_interest": 10.0,
                    "underlying_price": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_position_keeps_its_origin_exit_head_across_boundary() -> None:
    seen: list[tuple[str, str]] = []

    def policy(state: sim.DecisionState) -> sim.Action:
        if state.minute in ("12:45", "12:46", "13:00"):
            seen.append((state.minute, state.role))
        if state.minute == "12:45":
            return sim.Action("BUY", "call")
        if state.minute == "13:00":
            return sim.Action("SELL")
        return sim.Action("HOLD" if state.position else "ABSTAIN")

    result = sim.simulate_session(_quotes(), SESSION, policy, trade_cap=1)
    assert ("12:45", "morning_entry") in seen
    assert ("12:46", "morning_exit") in seen
    assert ("13:00", "morning_exit") in seen
    assert result.trades.iloc[0]["origin_regime"] == "morning"


def test_pending_exit_fills_first_later_bid_not_best_later_bid() -> None:
    quotes = _quotes(missing_bid_minutes={"10:00"})
    quotes.loc[(quotes.contract_id.eq("call")) & quotes.event_time.dt.tz_convert("America/New_York").dt.strftime("%H:%M").eq("10:02"), ["bid", "ask", "mid"]] = [20.0, 20.2, 20.1]
    ledger = sim.ActionLedger(
        {
            ("09:35", "morning_entry"): sim.Action("BUY", "call"),
            ("10:00", "morning_exit"): sim.Action("SELL", reason="test"),
        }
    )
    result = sim.simulate_session(quotes, SESSION, ledger, trade_cap=1)
    trade = result.trades.iloc[0]
    assert trade["requested_exit_minute"] == "10:00"
    assert trade["exit_minute"] == "10:01"
    assert trade["exit_bid"] < 20.0


def test_terminal_without_bid_or_validated_settlement_is_blocked() -> None:
    quotes = _quotes(missing_bid_minutes={"16:00"})
    ledger = sim.ActionLedger({("15:00", "afternoon_entry"): sim.Action("BUY", "call")})
    result = sim.simulate_session(quotes, SESSION, ledger, trade_cap=1)
    assert result.blocked_terminal_position
    assert result.ending_cash_usd is None
    assert result.trades.empty


def test_validated_terminal_settlement_uses_intrinsic_and_labels_it() -> None:
    quotes = _quotes(missing_bid_minutes={"16:00"})
    ledger = sim.ActionLedger({("15:00", "afternoon_entry"): sim.Action("BUY", "call")})
    result = sim.simulate_session(
        quotes, SESSION, ledger, trade_cap=1, validated_settlement_spx=110.0
    )
    trade = result.trades.iloc[0]
    assert trade["exit_type"] == "validated_cash_settlement"
    assert trade["exit_proceeds_after_fee_usd"] == pytest.approx(500.0 - 3.08)


def test_zero_recovery_is_explicitly_a_sensitivity_not_a_bid() -> None:
    quotes = _quotes(missing_bid_minutes={"16:00"})
    ledger = sim.ActionLedger({("15:00", "afternoon_entry"): sim.Action("BUY", "call")})
    result = sim.simulate_session(
        quotes, SESSION, ledger, trade_cap=1, terminal_zero_recovery=True
    )
    trade = result.trades.iloc[0]
    assert trade["exit_type"] == "zero_recovery_sensitivity"
    assert np.isnan(trade["exit_bid"])


def test_fill_accounting_is_ask_in_bid_out_fee_once() -> None:
    ledger = sim.ActionLedger(
        {
            ("09:35", "morning_entry"): sim.Action("BUY", "call"),
            ("09:36", "morning_exit"): sim.Action("SELL"),
        }
    )
    result = sim.simulate_session(_quotes(), SESSION, ledger, trade_cap=1)
    trade = result.trades.iloc[0]
    expected = (trade["exit_bid"] - trade["entry_ask"]) * 100.0 - 3.08
    assert trade["net_pnl_usd"] == pytest.approx(expected)
    assert result.ending_cash_usd == pytest.approx(10_000.0 + expected)


def test_trade_ledger_tracks_otm_to_itm_path_and_underlying_excursions() -> None:
    quotes = _quotes()
    is_0936 = (
        quotes["event_time"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
        .eq("09:36")
    )
    quotes.loc[is_0936, "underlying_price"] = 110.0
    ledger = sim.ActionLedger(
        {
            ("09:35", "morning_entry"): sim.Action("BUY", "call"),
            ("09:36", "morning_exit"): sim.Action("SELL"),
        }
    )
    trade = sim.simulate_session(quotes, SESSION, ledger, trade_cap=1).trades.iloc[0]
    assert trade["entry_moneyness_itm_points"] == -5.0
    assert trade["maximum_itm_depth_points"] == 5.0
    assert trade["final_itm_depth_points"] == 5.0
    assert trade["otm_to_itm_conversion"]
    assert trade["time_to_cross_minutes"] == 1
    assert trade["underlying_mfe_points"] == 10.0
    assert trade["underlying_mae_points"] == 0.0


def test_one_position_rejects_overlapping_buy() -> None:
    def policy(state: sim.DecisionState) -> sim.Action:
        if state.minute == "09:35":
            return sim.Action("BUY", "call")
        if state.minute == "09:36":
            return sim.Action("BUY", "put")
        if state.minute == "09:37":
            return sim.Action("SELL")
        return sim.Action("HOLD" if state.position else "ABSTAIN")

    result = sim.simulate_session(_quotes(), SESSION, policy, trade_cap=2)
    rejected = result.events[result.events["status"].eq("rejected_overlapping_position")]
    assert len(rejected) == 1
    assert len(result.trades) == 1


def test_trade_cap_rejects_second_entry() -> None:
    ledger = sim.ActionLedger(
        {
            ("09:35", "morning_entry"): sim.Action("BUY", "call"),
            ("09:36", "morning_exit"): sim.Action("SELL"),
            ("09:37", "morning_entry"): sim.Action("BUY", "put"),
        }
    )
    result = sim.simulate_session(_quotes(), SESSION, ledger, trade_cap=1)
    assert result.events["status"].eq("rejected_trade_cap").any()


def test_maximum_hold_forces_exit_without_consulting_policy() -> None:
    ledger = sim.ActionLedger({("09:35", "morning_entry"): sim.Action("BUY", "call")})
    result = sim.simulate_session(_quotes(), SESSION, ledger, trade_cap=1)
    trade = result.trades.iloc[0]
    assert trade["minutes_held"] == 120
    assert trade["exit_reason"] == "maximum_hold"


def test_future_quote_mutation_cannot_change_an_earlier_trade() -> None:
    clean = _quotes()
    dirty = clean.copy()
    future = dirty["event_time"].dt.tz_convert("America/New_York").dt.strftime("%H:%M").gt("10:00")
    dirty.loc[future, ["bid", "ask", "mid"]] *= 10.0
    ledger = sim.ActionLedger(
        {
            ("09:35", "morning_entry"): sim.Action("BUY", "call"),
            ("09:36", "morning_exit"): sim.Action("SELL"),
        }
    )
    a = sim.simulate_session(clean, SESSION, ledger, trade_cap=1)
    b = sim.simulate_session(dirty, SESSION, ledger, trade_cap=1)
    pd.testing.assert_frame_equal(a.trades, b.trades)


def test_policy_sees_whole_ladder_and_held_contract_separately() -> None:
    quotes = _quotes()
    quotes.loc[quotes["contract_id"].eq("call"), "underlying_price"] = 100.0
    seen: dict[str, object] = {}

    def policy(state: sim.DecisionState) -> sim.Action:
        if state.minute == "09:35":
            seen["entry_ladder_money"] = state.ladder_snapshot[
                "moneyness_itm_points"
            ].tolist()
            return sim.Action("BUY", "call")
        if state.minute == "09:36":
            seen["position_quote"] = state.position_quote
            return sim.Action("SELL")
        return sim.Action("HOLD" if state.position else "ABSTAIN")

    sim.simulate_session(quotes, SESSION, policy, trade_cap=1)
    assert any(abs(value) > 25.0 for value in seen["entry_ladder_money"])
    assert seen["position_quote"] is not None


def test_replay_capture_records_every_live_contract_and_probabilities() -> None:
    def policy(state: sim.DecisionState) -> sim.Action:
        if state.minute == "09:35":
            ids = state.entry_candidates["contract_id"].astype(str).tolist()
            probabilities = {contract_id: 0.4 for contract_id in ids}
            return sim.Action(
                "BUY",
                "call",
                reason="fixture",
                probability=0.4,
                candidate_probabilities=probabilities,
                diagnostics={"setup": "opening_turn"},
            )
        if state.minute == "09:36":
            return sim.Action("SELL", probability=0.6)
        return sim.Action("HOLD" if state.position else "ABSTAIN")

    result = sim.simulate_session(
        _quotes(), SESSION, policy, trade_cap=1, capture_replay_state=True
    )
    at_entry = result.considered_ladder[
        result.considered_ladder["minute"].eq("09:35")
    ]
    assert len(at_entry) == 3
    assert at_entry["selected"].sum() == 1
    assert not at_entry.loc[at_entry["contract_id"].eq("deep_call"), "entry_eligible"].iloc[0]
    assert result.events.loc[
        result.events["minute"].eq("09:35"), "policy_diagnostics_json"
    ].iloc[0] == '{"setup": "opening_turn"}'


def test_rejects_probability_for_an_ineligible_contract() -> None:
    def policy(state: sim.DecisionState) -> sim.Action:
        return sim.Action(
            "ABSTAIN", candidate_probabilities={"not-visible": 0.5}
        )

    with pytest.raises(sim.SimulatorError, match="ineligible contracts"):
        sim.simulate_session(_quotes(), SESSION, policy, trade_cap=1)


def test_stateful_policy_is_reset_at_each_episode_start() -> None:
    class StatefulPolicy:
        carries_session_state = True

        def __init__(self) -> None:
            self.reset_calls: list[str] = []
            self.counter = 99

        def reset_session(self, session: str) -> None:
            self.reset_calls.append(session)
            self.counter = 0

        def __call__(self, state: sim.DecisionState) -> sim.Action:
            self.counter += 1
            return sim.Action("ABSTAIN")

    policy = StatefulPolicy()
    sim.simulate_session(_quotes(), SESSION, policy, trade_cap=1)
    first_count = policy.counter
    sim.simulate_session(_quotes(), SESSION, policy, trade_cap=1)
    assert policy.reset_calls == [SESSION, SESSION]
    assert policy.counter == first_count


def test_stateful_policy_without_reset_contract_is_refused() -> None:
    class BrokenPolicy:
        carries_session_state = True

        def __call__(self, state: sim.DecisionState) -> sim.Action:
            return sim.Action("ABSTAIN")

    with pytest.raises(sim.SimulatorError, match="reset_session"):
        sim.simulate_session(_quotes(), SESSION, BrokenPolicy(), trade_cap=1)
