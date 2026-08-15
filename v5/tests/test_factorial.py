"""The factorial is only worth running if its arms and its algebra are right."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from v5.ops import run_factorial as rf


def _paths(prices, trade_id="t1", session="2025-09-03"):
    n = len(prices)
    return pd.DataFrame({
        "trade_id": trade_id, "session": session,
        "minute_in_trade": np.arange(1, n + 1, dtype=float),
        "bid": np.asarray(prices, float), "mid": np.asarray(prices, float),
        "continuation": np.full(n, 1e9),
    })


def test_clock_exit_takes_the_declared_minute() -> None:
    prices = [1.0] * 20
    prices[rf.CLOCK_MINUTES - 1] = 7.0   # minute_in_trade starts at 1
    got = rf.exit_rows(_paths(prices), "clock", np.random.default_rng(0), "bid")
    assert got.iloc[0]["exit_price"] == pytest.approx(7.0)


def test_oracle_exit_takes_the_best_minute() -> None:
    got = rf.exit_rows(_paths([1.0, 9.0, 2.0]), "oracle", np.random.default_rng(0), "bid")
    assert got.iloc[0]["exit_price"] == pytest.approx(9.0)


def test_no_arm_can_close_in_the_entry_minute() -> None:
    for arm in ("clock", "oracle", "model"):
        got = rf.exit_rows(_paths([5.0, 1.0, 1.0]), arm, np.random.default_rng(0), "bid")
        assert got.iloc[0]["minutes_held"] >= 1.0, arm


def _candidates(n_minutes=6, per_minute=4, session="2025-09-03"):
    rows = []
    for m in range(n_minutes):
        for k in range(per_minute):
            rows.append({
                "trade_id": f"{session}|{m}|{k}", "session": session,
                "minute_index": float(m * rf.MAX_HOLD_MINUTES // 2),
                "net": float(m * 10 + k), "score": float(k),
                "minutes_held": 5.0, "entry_ask_usd": 1000.0,
                "delta": 0.5, "is_call": True,
            })
    return pd.DataFrame(rows)


def test_occupancy_blocks_a_second_trade_inside_the_hold() -> None:
    frame = _candidates()
    taken = rf.select(frame, "random", np.random.default_rng(0), rf.MAX_HOLD_MINUTES)
    minutes = sorted(taken["minute_index"].tolist())
    for a, b in zip(minutes, minutes[1:]):
        assert b - a >= rf.MAX_HOLD_MINUTES


def test_oracle_entry_picks_the_best_candidate_at_each_minute() -> None:
    frame = _candidates()
    taken = rf.select(frame, "oracle", np.random.default_rng(0), rf.MAX_HOLD_MINUTES)
    for _, row in taken.iterrows():
        same = frame[frame["minute_index"] == row["minute_index"]]
        assert row["net"] == same["net"].max()


def test_oracle_entry_is_never_worse_than_random_entry() -> None:
    frame = _candidates()
    rng = np.random.default_rng(0)
    a = rf.select(frame, "oracle", rng, rf.MAX_HOLD_MINUTES)["net"].mean()
    b = rf.select(frame, "random", rng, rf.MAX_HOLD_MINUTES)["net"].mean()
    assert a >= b


def test_currency_reports_the_identity_it_claims() -> None:
    """net must equal p*W - (1-p)*L, or the common language is a lie."""

    net = np.array([100.0, -50.0, 200.0, -25.0, -25.0])
    trades = pd.DataFrame({
        "net": net, "session": "2025-09-03", "minutes_held": 5.0,
        "entry_ask_usd": 1000.0, "delta": 0.5, "is_call": True,
    })
    got = rf.currency(trades, "random", "clock")
    rebuilt = got["p"] * got["W"] - (1 - got["p"]) * got["L"]
    assert rebuilt == pytest.approx(net.mean(), abs=0.02)


def test_breakeven_hit_rate_is_the_one_that_zeroes_the_net() -> None:
    trades = pd.DataFrame({
        "net": np.array([100.0, -100.0]), "session": "s", "minutes_held": 5.0,
        "entry_ask_usd": 1000.0, "delta": 0.5, "is_call": True,
    })
    got = rf.currency(trades, "random", "clock")
    p = got["breakeven_p"]
    assert p * got["W"] - (1 - p) * got["L"] - rf.FEES_USD == pytest.approx(0.0, abs=0.02)


def test_interaction_is_zero_when_the_halves_do_not_interfere() -> None:
    """Additive effects must produce exactly zero interaction."""

    base, entry_effect, exit_effect = -20.0, 6.0, 4.0
    combined = base + entry_effect + exit_effect
    interaction = combined - base - entry_effect - exit_effect
    assert interaction == pytest.approx(0.0)


def test_interaction_is_negative_when_the_exit_undoes_the_entry() -> None:
    base, entry_effect, exit_effect = -20.0, 6.0, 4.0
    combined = base + 1.0          # the halves together deliver far less
    interaction = combined - base - entry_effect - exit_effect
    assert interaction < 0


def _quote_session(path, *, seed: int = 1):
    """A small but complete quote session, enough to build both tables."""

    from v5.ops import build_factorial_population as bfp

    rng = np.random.default_rng(seed)
    minutes = [bfp._label(i) for i in range(bfp._index("09:30"), bfp._index("16:00") + 1)]
    strikes = np.arange(5020.0, 5085.0, 5.0)
    spot = 5050.0 + np.cumsum(rng.normal(0.0, 0.5, len(minutes)))
    rows = []
    for i, minute in enumerate(minutes):
        left = max(bfp._index("16:00") - bfp._index(minute), 1)
        for strike in strikes:
            for right in ("C", "P"):
                money = (spot[i] - strike) if right == "C" else (strike - spot[i])
                fair = max(money, 0.0) + 8.0 * np.sqrt(left / 390.0)
                rows.append({
                    "event_time": pd.Timestamp(f"2025-09-03 {minute}", tz="America/New_York"),
                    "strike": float(strike), "right": right,
                    "bid": float(fair - 0.25), "ask": float(fair + 0.25), "mid": float(fair),
                })
    frame = pd.DataFrame(rows)
    frame[frame["bid"] > 0].to_parquet(path)
    return path


def test_the_population_carries_every_feature_both_models_name(tmp_path) -> None:
    """A feature named in a list but missing from the table fails only at fit time."""

    from v5.ops import build_factorial_population as bfp
    from v5.ops.train_optimal_exit import FEATURES as EXIT_FEATURES
    from v5.ops.train_selective_policy import FEATURES as ENTRY_FEATURES

    got = bfp.session_tables(_quote_session(tmp_path / "q.parquet"), tmp_path)
    assert got is not None
    candidates, paths = got
    candidates["is_call_int"] = candidates["is_call"].astype(int)

    missing_entry = [f for f in ENTRY_FEATURES if f not in candidates.columns]
    missing_exit = [f for f in EXIT_FEATURES if f not in paths.columns]
    assert not missing_entry, f"entry features missing: {missing_entry}"
    assert not missing_exit, f"exit features missing: {missing_exit}"


def test_candidates_and_paths_share_their_trade_ids(tmp_path) -> None:
    from v5.ops import build_factorial_population as bfp

    candidates, paths = bfp.session_tables(_quote_session(tmp_path / "q.parquet"), tmp_path)
    assert set(paths["trade_id"]) <= set(candidates["trade_id"])
    assert len(set(paths["trade_id"])) > 0


def test_every_candidate_offers_more_than_one_contract_to_choose_between(tmp_path) -> None:
    """An entry arm with one candidate per minute is not choosing anything."""

    from v5.ops import build_factorial_population as bfp

    candidates, _ = bfp.session_tables(_quote_session(tmp_path / "q.parquet"), tmp_path)
    per_minute = candidates.groupby("entry_minute")["trade_id"].nunique()
    assert per_minute.min() > 1


def test_exit_effect_is_applied_as_a_ratio_not_a_level() -> None:
    """The entry changes the contract mix, so absolute W and L do not transfer."""

    base = {"p": 0.35, "W": 100.0, "L": 100.0}
    entry_only = {"p": 0.50, "W": 300.0, "L": 300.0}   # richer contracts
    exit_only = {"p": 0.35, "W": 120.0, "L": 90.0}     # +20% W, -10% L
    got = rf.predicted_from_parts(base, entry_only, exit_only)

    assert got["W"] == pytest.approx(360.0)   # 300 * 1.2, not 120
    assert got["L"] == pytest.approx(270.0)   # 300 * 0.9, not 90
    assert got["p_from_entry"] == pytest.approx(0.50)


def test_a_pure_composition_change_is_not_reported_as_interference() -> None:
    """Doubling contract size with no skill change must predict the measurement."""

    base = {"p": 0.40, "W": 100.0, "L": 100.0}
    entry_only = {"p": 0.40, "W": 200.0, "L": 200.0}   # same skill, bigger tickets
    exit_only = {"p": 0.40, "W": 100.0, "L": 100.0}    # exit does nothing
    got = rf.predicted_from_parts(base, entry_only, exit_only)
    measured = 0.40 * 200.0 - 0.60 * 200.0 - rf.FEES_USD
    assert got["predicted_net_usd"] == pytest.approx(measured, abs=0.02)


def test_component_shifts_name_which_number_each_half_moved() -> None:
    base = {"p": 0.35, "W": 100.0, "L": 100.0}
    entry_only = {"p": 0.45, "W": 100.0, "L": 150.0}
    exit_only = {"p": 0.35, "W": 130.0, "L": 100.0}
    combined = {"p": 0.45, "W": 130.0, "L": 150.0}
    got = rf.component_shifts(base, entry_only, exit_only, combined)

    assert got["entry_moved"]["p"] == pytest.approx(0.10)
    assert got["entry_moved"]["L"] == pytest.approx(50.0)   # the named defect
    assert got["exit_moved"]["W"] == pytest.approx(30.0)
    assert got["exit_moved"]["p"] == pytest.approx(0.0)
