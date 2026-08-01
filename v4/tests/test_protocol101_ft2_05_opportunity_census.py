from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v4.scripts.run_protocol101_ft2_05_opportunity_census import (
    ONE_MINUTE_NS,
    QuotePath,
    add_quality_scores,
    compute_horizon_labels,
    friction_table,
    guardrail_curve_tables,
    outcome_bucket,
    pareto_frontier_table,
    power_tables,
    replay_variant,
    verify_authority_and_inputs,
)


def test_frozen_authority_and_inputs_match() -> None:
    receipt, label_spec, oracle_rules, census = verify_authority_and_inputs()
    assert receipt["role_counts"]["census"] == 45
    assert label_spec["horizons_minutes"] == [
        3,
        5,
        10,
        20,
        45,
        90,
        "remaining_session",
    ]
    assert oracle_rules["rule_count"] == 3
    assert len(census["census_sessions"]) == 45


def test_path_labels_use_tplus1_ask_bid_fee_and_no_bid_full_loss() -> None:
    decision = pd.Timestamp("2025-01-02 09:30", tz="America/New_York")
    decision_ns = int(decision.tz_convert("UTC").value)
    path = QuotePath(
        quote_ns=np.asarray(
            [
                decision_ns + ONE_MINUTE_NS,
                decision_ns + 2 * ONE_MINUTE_NS,
                decision_ns + 3 * ONE_MINUTE_NS,
            ],
            dtype=np.int64,
        ),
        bid=np.asarray([0.90, 0.0, 1.10]),
        ask=np.asarray([1.00, 0.05, 1.20]),
        quote_age_ms=np.zeros(3),
    )
    labels = compute_horizon_labels(
        path=path,
        entry_fill_ns=decision_ns + ONE_MINUTE_NS,
        session="2025-01-02",
        entry_ask=1.0,
        horizon=2,
    )
    assert labels["available_minutes"] == 2
    assert labels["executable_bid_minutes"] == 1
    assert labels["no_bid_minutes"] == 1
    assert labels["mfe_dollars"] == pytest.approx(7.0)
    assert labels["ttfp_minutes"] == 2.0
    assert labels["ppae_dollars"] == pytest.approx(-103.0)
    assert labels["underwater_minutes"] == 1
    assert labels["longest_profitable_minutes"] == 1
    assert labels["profitable_window_count"] == 1
    assert labels["fraction_positive"] == 0.5


def test_d48_d49_selection_replays_through_simulator_v5() -> None:
    decision = pd.Timestamp("2025-01-02 09:32", tz="America/New_York")
    decision_ns = int(decision.tz_convert("UTC").value)
    deadline_ns = decision_ns + 3 * ONE_MINUTE_NS
    rows = []
    for right, offset, pnl, ask in (
        ("C", 0.0, 97.0, 4.0),
        ("P", 0.0, 47.0, 4.0),
        ("C", 5.0, 197.0, 6.0),
    ):
        exit_bid = ask + (pnl + 3.0) / 100.0
        rows.append(
            {
                "session": "2025-01-02",
                "month": "2025-01",
                "vol_regime": "low",
                "decision_time_ns": decision_ns,
                "contract_id": f"test-{right}-{offset}",
                "right": right,
                "strike_idx": int(offset / 5) + 10,
                "offset": offset,
                "premium_band": "medium_3_8",
                "moneyness_band": "atm",
                "decision_entry_ask": ask,
                "decision_entry_ask_cents": int(round(ask * 100)),
                "decision_quote_base_eligible": True,
                "intent_cost_fee3_cents": int(round(ask * 10_000)) + 300,
                "intent_eligible_fee3_at_t": ask <= 4.97,
                "entry_ask": ask,
                "entry_ask_cents": int(round(ask * 100)),
                "fill_cost_fee3_cents": int(round(ask * 10_000)) + 300,
                "fill_recheck_pass_fee3_at_tplus1": ask <= 4.97,
                "premium_plus_fee": ask * 100.0 + 3.0,
                "vwap_side": "C",
                "best_exit_time_ns": deadline_ns,
                "best_source_time_ns": deadline_ns,
                "h3_best_exit_time_ns": deadline_ns,
                "h3_deadline_ns": deadline_ns,
                "h3_best_exit_bid": exit_bid,
                "h3_mfe_dollars": pnl,
                "conservative_upside_return": pnl / (ask * 100.0),
            }
        )
    frame = pd.DataFrame(rows)
    sessions, payload = replay_variant(
        frame,
        variant="best_3",
        selector="oracle",
    )
    # The $6 contract has the highest hindsight PnL but breaches D48 at $10k.
    assert payload["summary"]["trade_count"] == 1
    assert payload["summary"]["pooled_pnl"] == 97.0
    assert sessions["trade_count"].sum() == 1


def test_oracle_commits_before_selected_only_fill_recheck_without_substitute() -> None:
    decision = pd.Timestamp("2025-01-02 09:32", tz="America/New_York")
    decision_ns = int(decision.tz_convert("UTC").value)
    deadline_ns = decision_ns + 3 * ONE_MINUTE_NS
    rows = []
    for contract_id, pnl, fill_ask in (
        ("future-best-rejects", 197.0, 6.0),
        ("lower-but-fill-passes", 97.0, 4.0),
    ):
        decision_ask = 4.0
        exit_bid = fill_ask + (pnl + 3.0) / 100.0
        rows.append(
            {
                "session": "2025-01-02",
                "month": "2025-01",
                "vol_regime": "low",
                "decision_time_ns": decision_ns,
                "contract_id": contract_id,
                "right": "C",
                "strike_idx": 10,
                "offset": 0.0 if contract_id.startswith("future") else 5.0,
                "premium_band": "medium_3_8",
                "moneyness_band": "atm",
                "decision_entry_ask": decision_ask,
                "decision_entry_ask_cents": 400,
                "decision_quote_base_eligible": True,
                "intent_cost_fee3_cents": 40_300,
                "intent_eligible_fee3_at_t": True,
                "entry_ask": fill_ask,
                "entry_ask_cents": int(round(fill_ask * 100)),
                "fill_cost_fee3_cents": int(round(fill_ask * 10_000)) + 300,
                "fill_recheck_pass_fee3_at_tplus1": fill_ask <= 4.97,
                "premium_plus_fee": fill_ask * 100.0 + 3.0,
                "vwap_side": "C",
                "best_exit_time_ns": deadline_ns,
                "best_source_time_ns": deadline_ns,
                "h3_best_exit_time_ns": deadline_ns,
                "h3_deadline_ns": deadline_ns,
                "h3_best_exit_bid": exit_bid,
                "h3_mfe_dollars": pnl,
                "conservative_upside_return": pnl / (fill_ask * 100.0),
            }
        )
    sessions, payload = replay_variant(
        pd.DataFrame(rows),
        variant="best_3",
        selector="oracle",
    )
    assert payload["summary"]["trade_count"] == 0
    assert payload["summary"]["pooled_pnl"] == 0.0
    assert payload["summary"]["rejected_fill_count"] == 1
    assert sessions["trade_count"].sum() == 0
    rejected = payload["rejected_fills"]
    assert len(rejected) == 1
    event = rejected.iloc[0]
    assert event["source_neutral_contract_id"] == "future-best-rejects"
    assert bool(event["position_opened"]) is False
    assert event["premium_charged_cents"] == 0
    assert event["fee_charged_cents"] == 0


def test_four_bucket_boundaries_are_fee_aware() -> None:
    assert outcome_bucket(0.40) == "big_win"
    assert outcome_bucket(0.05) == "scratch_or_small_win"
    assert outcome_bucket(-0.05) == "scratch_or_small_win"
    assert outcome_bucket(-0.10) == "small_loss"
    assert outcome_bucket(-0.30) == "big_loss"


def test_friction_table_discloses_intent_fill_and_valid_populations() -> None:
    frame = pd.DataFrame(
        [
            {
                "d48_reference_eligible": True,
                "premium_band": "small_1_3",
                "month": "2025-01",
                "vol_regime": "low",
                "moneyness_band": "atm",
                "fill_recheck_pass_fee3_at_tplus1": True,
                "friction_fraction_of_premium": 0.05,
                "entry_spread": 0.10,
            },
            {
                "d48_reference_eligible": True,
                "premium_band": "small_1_3",
                "month": "2025-01",
                "vol_regime": "low",
                "moneyness_band": "atm",
                "fill_recheck_pass_fee3_at_tplus1": False,
                "friction_fraction_of_premium": np.nan,
                "entry_spread": 0.20,
            },
        ]
    )
    table = friction_table(frame)
    overall = table[table["stratification"] == "premium_band"].iloc[0]
    assert overall["contract_minutes"] == 2
    assert overall["intent_contract_minutes"] == 2
    assert overall["fill_pass_rows"] == 1
    assert overall["rejected_fill_rows"] == 1
    assert overall["friction_valid_rows"] == 1
    assert overall["entry_spread_valid_rows"] == 2


def test_headline_curves_are_stratified_without_rethresholding() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "session": f"2025-0{1 + index // 4}-{2 + index:02d}",
                "month": "2025-01" if index < 4 else "2025-02",
                "vol_regime": "low" if index % 2 == 0 else "high",
                "decision_time_ns": index * ONE_MINUTE_NS,
                "decision_time_utc": f"2025-01-02T14:{30 + index:02d}:00+00:00",
                "contract_id": f"test-{index}",
                "right": "C" if index % 2 == 0 else "P",
                "offset": float(index * 5),
                "market_phase": "primary_morning",
                "premium_band": "small_1_3",
                "moneyness_band": "near",
                "d48_reference_eligible": True,
                "h10_early_dd_return": -0.01 * index,
                "session_ttfp_minutes": float(index + 1),
                "session_ppae_return": -0.02 * index,
                "session_uwi_return": float(index),
                "session_fraction_positive": 1.0 - 0.05 * index,
                "session_upside_q75_return": 0.10 + 0.01 * index,
                "session_mfe_return": 0.20 + 0.10 * index,
                "session_mfe_dollars": 20.0 + 10.0 * index,
            }
        )
    frame = add_quality_scores(pd.DataFrame(rows))
    curves, excluded, _ = guardrail_curve_tables(frame)
    expected_regimes = {
        ("overall", "all"),
        ("month", "2025-01"),
        ("month", "2025-02"),
        ("vol_regime", "low"),
        ("vol_regime", "high"),
    }
    assert set(zip(curves.regime_dimension, curves.regime_value)) == expected_regimes
    assert set(zip(excluded.regime_dimension, excluded.regime_value)) == expected_regimes

    frontier = pareto_frontier_table(frame)
    assert set(zip(frontier.regime_dimension, frontier.regime_value)) == expected_regimes

    session_rows = []
    trade_rows = []
    for row in rows:
        for selector, pnl in (("oracle", 100.0), ("p5", 25.0)):
            session_rows.append(
                {
                    "session": row["session"],
                    "month": row["month"],
                    "vol_regime": row["vol_regime"],
                    "selector": selector,
                    "variant": "best_session",
                    "pnl": pnl,
                    "trade_count": 1,
                }
            )
            trade_rows.append(
                {
                    "session": row["session"],
                    "month": row["month"],
                    "vol_regime": row["vol_regime"],
                    "selector": selector,
                    "variant": "best_session",
                    "pnl": pnl,
                }
            )
    variance, mde = power_tables(
        pd.DataFrame(session_rows),
        pd.DataFrame(trade_rows),
    )
    assert set(zip(variance.regime_dimension, variance.regime_value)) == expected_regimes
    assert set(zip(mde.regime_dimension, mde.regime_value)) == expected_regimes
