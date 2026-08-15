from __future__ import annotations

import math

import numpy as np

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES, OPTION_FEATURE_NAMES
from v4.scripts.run_protocol164_full_action_space_dataset import processed_rows_to_candidates


def test_protocol164_uses_full_midday_ladder_without_protocol101_gate() -> None:
    rows = [_processed_row("2026-01-02T17:00:00+00:00")]

    candidates = processed_rows_to_candidates(
        split="unit",
        session="2026-01-02",
        rows=rows,
        starting_cash=10_000.0,
    )

    assert len(candidates) == 2
    assert {row["contract_id"] for row in candidates} == {
        "SPXW-20260102-04000.000-C",
        "SPXW-20260102-04000.000-P",
    }
    assert {row["time_bucket"] for row in candidates} == {"midday"}
    assert all(row["root"] == "SPXW" for row in candidates)
    assert all(row["settlement_style"] == "PM" for row in candidates)


def test_protocol164_repairs_missing_greeks_before_skipping_candidate() -> None:
    rows = [_processed_row("2026-01-02T15:00:00+00:00")]

    candidates = processed_rows_to_candidates(
        split="unit",
        session="2026-01-02",
        rows=rows,
        starting_cash=10_000.0,
    )
    put = next(row for row in candidates if row["right"] == "P")

    assert put["contract_id"] == "SPXW-20260102-04000.000-P"
    assert math.isfinite(float(put["entry_delta"]))
    assert math.isfinite(float(put["entry_gamma"]))
    assert math.isfinite(float(put["entry_theta"]))
    assert put["entry_delta"] == 0.0
    assert put["entry_gamma"] == 0.0
    assert put["entry_theta"] == 0.0


def test_protocol164_masks_unaffordable_but_keeps_candidate_for_training() -> None:
    rows = [_processed_row("2026-01-02T15:00:00+00:00", ask=125.0)]

    candidates = processed_rows_to_candidates(
        split="unit",
        session="2026-01-02",
        rows=rows,
        starting_cash=10_000.0,
    )

    assert len(candidates) == 2
    assert {row["entry_affordable_10k"] for row in candidates} == {0.0}


def test_protocol164_rejects_wrong_root_non_five_point_and_malformed_quotes() -> None:
    rows = [_processed_row("2026-01-02T15:00:00+00:00", bid=0.0)]

    candidates = processed_rows_to_candidates(
        split="unit",
        session="2026-01-02",
        rows=rows,
        starting_cash=10_000.0,
    )

    assert candidates == []


def _processed_row(decision_time: str, *, bid: float = 1.0, ask: float = 1.2) -> dict:
    option_index = {name: idx for idx, name in enumerate(OPTION_FEATURE_NAMES)}
    ladder = np.zeros((2, 2, len(OPTION_FEATURE_NAMES)), dtype=np.float32)
    for strike_idx in range(2):
        for right_idx in range(2):
            ladder[strike_idx, right_idx, option_index["bid"]] = bid
            ladder[strike_idx, right_idx, option_index["ask"]] = ask
            ladder[strike_idx, right_idx, option_index["mid"]] = (bid + ask) / 2.0
            ladder[strike_idx, right_idx, option_index["spread"]] = ask - bid
            ladder[strike_idx, right_idx, option_index["spread_frac"]] = 0.10
            ladder[strike_idx, right_idx, option_index["bid_size"]] = 10.0
            ladder[strike_idx, right_idx, option_index["ask_size"]] = 12.0
            ladder[strike_idx, right_idx, option_index["iv"]] = 0.20
            ladder[strike_idx, right_idx, option_index["delta"]] = 0.40 if right_idx == 0 else -0.40
            ladder[strike_idx, right_idx, option_index["gamma"]] = 0.012
            ladder[strike_idx, right_idx, option_index["theta"]] = -0.20
    ladder[0, 1, option_index["iv"]] = np.nan
    ladder[0, 1, option_index["delta"]] = np.nan
    ladder[0, 1, option_index["gamma"]] = np.nan
    ladder[0, 1, option_index["theta"]] = np.nan
    market = np.zeros((30, len(MARKET_FEATURE_NAMES)), dtype=np.float32)
    market[:, MARKET_FEATURE_NAMES.index("spx_close")] = 4000.0
    market[:, MARKET_FEATURE_NAMES.index("vix_close")] = 18.0
    return {
        "decision_time": decision_time,
        "strike_offsets": np.asarray([-5.0, 0.0], dtype=np.float32),
        "rights": ("C", "P"),
        "option_ladder": ladder,
        "candidate_mask": np.asarray([[True, True], [True, True]], dtype=bool),
        "contract_ids": np.asarray(
            [
                ["SPXW-20260102-04000.000-C", "SPXW-20260102-04000.000-P"],
                ["SPX-20260102-04005.000-C", "SPXW-20260102-04002.000-P"],
            ],
            dtype=object,
        ),
        "market_window": market,
    }
