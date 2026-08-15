"""The two-sided model: a fitted policy that may buy, sell or stand aside.

The load-bearing test is that a fold is never predicted by a model that saw it.
Everything else in this file is arithmetic; that one is the difference between a
result and a leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import train_two_sided_model as tsm


def _table(n_sessions: int, *, straddle_return: float = -0.01, seed: int = 1,
           signal: bool = False) -> pd.DataFrame:
    """Synthetic slots. With ``signal``, the label is readable from a feature."""

    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for slot in range(6):
            rel = rng.uniform(0.001, 0.01)
            ret = straddle_return + rng.normal(0, 0.05)
            if signal:
                # A clean, learnable relationship: wide recent range -> the
                # straddle pays; narrow -> it does not.
                ret = (0.25 if rel > 0.0055 else -0.25) + rng.normal(0, 0.02)
            premium = 2_000.0
            rows.append(
                {
                    "session": f"{2022 + s // 260}-{1 + (s // 28) % 12:02d}-{1 + s % 28:02d}",
                    "entry_minute": f"1{slot}:00",
                    "minutes_to_close": 385 - 60 * slot,
                    "hold": 60,
                    "spot": 5_000.0,
                    "abs_move": abs(rng.normal(0, 8)),
                    "range_15m": rel * 5_000.0 / 2,
                    "range_30m": rel * 5_000.0,
                    "range_60m": rel * 5_000.0 * 2,
                    "move_15m": rng.normal(0, 3),
                    "move_30m": rng.normal(0, 5),
                    "move_60m": rng.normal(0, 7),
                    "range_position": rng.uniform(),
                    "session_range": rel * 5_000.0 * 3,
                    "straddle_premium": premium,
                    "straddle_gross": ret * premium,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# features and label
# --------------------------------------------------------------------------


def test_every_declared_feature_is_produced() -> None:
    got = tsm.prepare(_table(60))
    for name in tsm.FEATURES:
        assert name in got, name
    assert tsm.LABEL in got


def test_features_are_scale_free_so_eras_are_comparable() -> None:
    """A 2022 session at a different index level must look the same."""

    base = _table(60, seed=4)
    scaled = base.copy()
    for col in ("spot", "range_15m", "range_30m", "range_60m", "move_15m",
                "move_30m", "move_60m", "session_range"):
        scaled[col] = scaled[col] * 2.0
    a, b = tsm.prepare(base), tsm.prepare(scaled)
    for name in ("range_30m_rel", "move_60m_rel", "session_range_rel"):
        assert np.allclose(a[name].to_numpy(), b[name].to_numpy())


def test_the_label_is_a_return_on_the_straddle() -> None:
    got = tsm.prepare(_table(60))
    assert got[tsm.LABEL].to_numpy() == pytest.approx(
        (got["straddle_gross"] / got["straddle_premium"]).to_numpy()
    )


# --------------------------------------------------------------------------
# the walk-forward must never see its own fold
# --------------------------------------------------------------------------


def test_a_fold_is_never_predicted_by_a_model_that_saw_it() -> None:
    """Plant a label only later sessions can reveal, and check it stays hidden.

    Early sessions carry no relationship; later ones carry a strong one. A model
    respecting chronology cannot predict the later block from the earlier, so
    its correlation on the first predicted fold must be near zero.
    """

    early = tsm.prepare(_table(300, straddle_return=0.0, seed=7))
    late = tsm.prepare(_table(200, seed=8, signal=True))
    late["session"] = "2099-" + late["session"].str[5:]
    table = pd.concat([early, late], ignore_index=True).sort_values(
        ["session", "entry_minute"]
    )
    got = tsm.walk_forward(table, shuffle_labels=False, seed=3)
    # Only the FIRST predicted fold is trained purely on the signal-free block.
    # Later folds have legitimately seen planted sessions, so including them
    # would test nothing.
    ordered = sorted(table["session"].unique())
    first_fold_ids = set(
        ordered[tsm.INITIAL_TRAIN_SESSIONS : tsm.INITIAL_TRAIN_SESSIONS + tsm.FOLD_SESSIONS]
    )
    first_fold = got[got["session"].isin(first_fold_ids)]
    assert len(first_fold) > 0
    assert all(s.startswith("2099") for s in first_fold["session"])
    corr = np.corrcoef(first_fold["predicted_return"], first_fold[tsm.LABEL])[0, 1]
    assert abs(corr) < 0.2

    # And the very next fold, which may see the planted block, does learn it —
    # so the assertion above is about chronology, not about a dead pipeline.
    second_fold_ids = set(
        ordered[
            tsm.INITIAL_TRAIN_SESSIONS + tsm.FOLD_SESSIONS :
            tsm.INITIAL_TRAIN_SESSIONS + 2 * tsm.FOLD_SESSIONS
        ]
    )
    second = got[got["session"].isin(second_fold_ids)]
    assert np.corrcoef(second["predicted_return"], second[tsm.LABEL])[0, 1] > 0.4


def test_a_relationship_present_throughout_is_learned() -> None:
    """The recovery control: the pipeline must find a real signal."""

    table = tsm.prepare(_table(600, seed=9, signal=True))
    got = tsm.walk_forward(table, shuffle_labels=False, seed=3)
    corr = np.corrcoef(got["predicted_return"], got[tsm.LABEL])[0, 1]
    assert corr > 0.5


def test_shuffling_the_labels_destroys_the_relationship() -> None:
    table = tsm.prepare(_table(600, seed=9, signal=True))
    got = tsm.walk_forward(table, shuffle_labels=True, seed=3)
    corr = np.corrcoef(got["predicted_return"], got[tsm.LABEL])[0, 1]
    assert abs(corr) < 0.2


def test_the_first_training_block_is_never_scored() -> None:
    table = tsm.prepare(_table(500, seed=11))
    got = tsm.walk_forward(table, shuffle_labels=False, seed=3)
    scored = set(got["session"])
    earliest = sorted(table["session"].unique())[: tsm.INITIAL_TRAIN_SESSIONS]
    assert not scored & set(earliest)


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------


def _pred(edge_return: float, gross: float, n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": [f"2025-01-{1 + i % 28:02d}" for i in range(n)],
            "straddle_premium": 2_000.0,
            "straddle_gross": gross,
            "predicted_return": edge_return,
        }
    )


def test_the_policy_stands_aside_when_the_edge_does_not_clear_the_cost() -> None:
    # Predicted edge of $20 against a $50 round trip needing 1.5x.
    got = tsm.score_policy(_pred(0.01, -100.0), 25.0)
    assert got["trades"] == 0


def test_the_policy_sells_when_it_expects_the_straddle_to_fall() -> None:
    got = tsm.score_policy(_pred(-0.2, -400.0), 25.0)
    assert got["share_short"] == pytest.approx(1.0)
    # Short collects 400, pays two round trips.
    assert got["mean_net_usd"] == pytest.approx(400.0 - 50.0)


def test_the_policy_buys_when_it_expects_the_straddle_to_rise() -> None:
    got = tsm.score_policy(_pred(0.2, 400.0), 25.0)
    assert got["share_short"] == 0.0
    assert got["mean_net_usd"] == pytest.approx(400.0 - 50.0)


def test_a_wrong_side_loses_the_move_and_the_cost() -> None:
    got = tsm.score_policy(_pred(-0.2, 400.0), 25.0)
    assert got["mean_net_usd"] == pytest.approx(-400.0 - 50.0)


def test_randomising_the_side_keeps_the_selection_but_destroys_the_call() -> None:
    real = tsm.score_policy(_pred(-0.2, -400.0), 25.0)
    null = tsm.score_policy(_pred(-0.2, -400.0), 25.0, random_side_seed=1)
    assert null["trades"] == real["trades"]
    assert null["share_short"] == pytest.approx(0.5, abs=0.15)
    assert null["mean_net_usd"] < real["mean_net_usd"]


def test_the_tail_is_reported_alongside_the_mean() -> None:
    pred = _pred(-0.2, -400.0)
    pred.loc[0, "straddle_gross"] = 20_000.0  # one catastrophic short
    got = tsm.score_policy(pred, 25.0)
    assert got["worst_trade_usd"] == pytest.approx(-20_050.0)
    assert got["mean_net_usd"] < 350.0


def test_a_cheaper_round_trip_leaves_more_on_the_table() -> None:
    dear = tsm.score_policy(_pred(-0.2, -400.0), 25.0)
    cheap = tsm.score_policy(_pred(-0.2, -400.0), 3.08)
    assert cheap["mean_net_usd"] > dear["mean_net_usd"]
    assert cheap["mean_net_usd"] - dear["mean_net_usd"] == pytest.approx(
        2 * (25.0 - 3.08), abs=1e-6
    )
