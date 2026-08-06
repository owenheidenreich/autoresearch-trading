"""Loader and replay must be causal, complete, and identical for surrogates.

These tests use hand-built bars with known answers. Two integration tests touch
the real owned corpus, but only structurally — session counts, roll flags, and
causality — never a member's economics, because the surrogate known-answer gate
has not run yet and the frozen order of work forbids reading real outcomes
before it does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.research.direction import family, loader, replay


def _bars(
    session: str,
    *,
    instrument_id: int = 1,
    open_0930: float = 100.0,
    path: dict[str, float] | None = None,
    volume: float = 1000.0,
    minutes: int = 70,
) -> loader.SessionBars:
    """A session whose closes default to a flat line at ``open_0930``."""

    labels = []
    hour, minute = 9, 30
    for _ in range(minutes):
        labels.append(f"{hour:02d}:{minute:02d}")
        minute += 1
        if minute == 60:
            hour, minute = hour + 1, 0
    closes = np.full(len(labels), open_0930, dtype=float)
    for label, value in (path or {}).items():
        closes[labels.index(label)] = value
    opens = closes.copy()
    opens[0] = open_0930
    return loader.SessionBars(
        session=session,
        instrument_id=instrument_id,
        minute_et=tuple(labels),
        open=opens,
        high=closes + 1.0,
        low=closes - 1.0,
        close=closes,
        volume=np.full(len(labels), volume, dtype=float),
    )


def test_features_are_causal_and_use_only_earlier_sessions() -> None:
    sessions = [
        _bars("2026-01-02", open_0930=100.0, path={"09:34": 102.0}, volume=100.0),
        _bars("2026-01-05", open_0930=110.0, path={"09:34": 108.0}, volume=300.0),
        _bars("2026-01-06", open_0930=120.0, path={"09:34": 121.0}, volume=200.0),
    ]
    features = loader.session_features(sessions)

    assert features["first_five_minute_return"].tolist() == [2.0, -2.0, 1.0]
    # The volume baseline must never include its own session.
    assert np.isnan(features["volume_surprise"].iloc[0])
    assert features["prior_volume_median"].iloc[1] == 500.0  # 5 bars x 100
    assert features["prior_volume_median"].iloc[2] == 1000.0  # median(500, 1500)
    # The gap uses the previous session's FINAL close -- not its open, and not
    # its 09:34 close. In these fixtures the line is flat at open_0930 apart
    # from the 09:34 override, so the final closes are 100.0 and 110.0.
    assert np.isnan(features["overnight_gap"].iloc[0])
    assert features["prior_session_final_close"].tolist()[1:] == [100.0, 110.0]
    assert features["overnight_gap"].iloc[1] == 110.0 - 100.0
    assert features["overnight_gap"].iloc[2] == 120.0 - 110.0


def test_a_roll_boundary_is_flagged_and_excluded_from_gap_mechanisms() -> None:
    sessions = [
        _bars("2026-01-02", instrument_id=1),
        _bars("2026-01-05", instrument_id=1),
        _bars("2026-01-06", instrument_id=2),  # contract roll
        _bars("2026-01-07", instrument_id=2),
    ]
    features = loader.session_features(sessions)
    assert features["is_roll_boundary"].tolist() == [False, False, True, False]

    # M1 keeps every session; the gap mechanisms drop the first and the roll.
    assert loader.eligible_sessions(features, "M1").sum() == 4
    for mechanism in ("M3", "JOINT"):
        mask = loader.eligible_sessions(features, mechanism)
        assert mask.sum() == 2
        assert features.loc[mask, "session"].tolist() == ["2026-01-05", "2026-01-07"]


def test_exit_price_falls_back_to_the_close_when_the_horizon_runs_past_it() -> None:
    """Forced flat by the session close, per the frozen declaration."""

    short = _bars("2026-01-02", open_0930=100.0, path={"10:20": 130.0}, minutes=51)
    features = loader.session_features([short])
    # 09:35 + 60m = 10:35, which does not exist in a 51-bar session.
    assert features["exit_price_60m"].iloc[0] == 130.0
    assert features["exit_price_15m"].iloc[0] == 100.0  # 09:50 exists


def test_folds_are_contiguous_and_never_shorten_the_newest_block() -> None:
    labels = loader.chronological_folds(254, 5)
    assert len(labels) == 254
    assert sorted(set(labels.tolist())) == [0, 1, 2, 3, 4]
    assert (np.diff(labels) >= 0).all()  # contiguous, chronological
    counts = np.bincount(labels)
    assert counts[-1] >= counts[0] - 1  # the most recent fold is not the short one
    with pytest.raises(loader.LoaderError):
        loader.chronological_folds(3, 5)


# --- replay ------------------------------------------------------------------


def _features_for_replay(n: int = 10, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session": [f"2026-01-{day:02d}" for day in range(1, n + 1)],
            "instrument_id": 1,
            "first_five_minute_return": 1.0,
            "first_five_minute_volume": 100.0,
            "entry_price": 100.0,
            "prior_session_final_close": 99.0,
            "is_roll_boundary": False,
            "has_prior_session": True,
            "overnight_gap": 1.0,
            "prior_volume_median": 100.0,
            "volume_surprise": 1.0,
        }
    )
    for horizon in family.HORIZON_MINUTES:
        frame[f"exit_price_{horizon}m"] = 100.0
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _member(name: str) -> family.Member:
    return family.assert_member_registered(name)


def test_every_declared_session_produces_a_row_including_no_trade_days() -> None:
    features = _features_for_replay(n=254, volume_surprise=0.5)  # never occupied
    result = replay.replay_member(features, _member("M1.with.15m"))
    assert len(result) == 254
    assert result["trades"].sum() == 0
    assert (result["net_points"] == 0.0).all()


def test_friction_is_charged_once_per_completed_round_trip() -> None:
    features = _features_for_replay(n=254)
    features["exit_price_15m"] = 102.0  # +2 points with side +1
    result = replay.replay_member(features, _member("M1.with.15m"))
    traded = result[result.trades == 1]
    assert len(traded) == 254
    assert traded["gross_points"].unique().tolist() == [2.0]
    assert traded["net_points"].unique() == pytest.approx([2.0 - 0.358])


def test_the_against_direction_is_the_exact_sign_flip() -> None:
    features = _features_for_replay(n=254)
    features["exit_price_15m"] = 102.0
    with_ = replay.replay_member(features, _member("M1.with.15m"))
    against = replay.replay_member(features, _member("M1.against.15m"))
    assert against["side"].tolist() == (-with_["side"]).tolist()
    # Both pay friction; the flip is not a free lunch.
    assert (against["gross_points"] == -with_["gross_points"]).all()
    assert against["net_points"].iloc[0] == pytest.approx(-2.0 - 0.358)


def test_a_zero_score_is_no_position() -> None:
    """sign(0) = 0 follows from the declared side rule; it is not a tie-break."""

    features = _features_for_replay(n=254, first_five_minute_return=0.0)
    result = replay.replay_member(features, _member("M1.with.15m"))
    assert result["trades"].sum() == 0


def test_joint_trades_only_when_the_first_five_minutes_confirm_the_gap() -> None:
    features = _features_for_replay(n=254)
    features.loc[: len(features) // 2, "first_five_minute_return"] = -1.0  # disagrees
    mask = replay.occupancy_mask(features, "JOINT")
    assert mask.sum() == len(features) - (len(features) // 2 + 1)
    # M3 on the same rows ignores the five-minute sign entirely.
    assert replay.occupancy_mask(features, "M3").all()


def test_an_undeclared_member_cannot_be_replayed() -> None:
    features = _features_for_replay(n=254)
    rogue = family.Member(
        name="M1.with.45m",
        mechanism="M1",
        occupancy="whatever",
        score="whatever",
        side_rule=family.DIRECTIONS["with"],
        horizon_minutes=45,
        eligible_sessions=254,
    )
    with pytest.raises(ValueError, match="not in the frozen G1 family"):
        replay.replay_member(features, rogue)


def test_a_short_session_index_is_refused_rather_than_padded() -> None:
    features = _features_for_replay(n=100)
    with pytest.raises(replay.ReplayError, match="may never be dropped"):
        replay.replay_member(features, _member("M1.with.15m"))


def test_the_comparator_never_sees_the_fold_it_is_judged_against() -> None:
    features = _features_for_replay(n=254)
    # Long wins everywhere, so every fold after the first should choose it.
    features["exit_price_15m"] = 105.0
    _, chosen = replay.causal_comparator_net(features, _member("M1.with.15m"))
    assert chosen[0] == "no_trade"  # nothing earlier existed to choose from
    assert {chosen[fold] for fold in (1, 2, 3, 4)} == {"always_long"}


def test_the_comparator_uses_the_members_own_clock_and_friction() -> None:
    features = _features_for_replay(n=254)
    features["exit_price_15m"] = 102.0
    control = replay.constant_side_sessions(features, _member("M1.with.15m"), "always_long")
    assert control["net_points"].iloc[0] == pytest.approx(2.0 - 0.358)
    flat = replay.constant_side_sessions(features, _member("M1.with.15m"), "no_trade")
    assert (flat["net_points"] == 0.0).all()
    assert flat["trades"].sum() == 0


# --- integration against the owned corpus: structure only --------------------


def _corpus_available() -> bool:
    from pathlib import Path

    return Path(family.ES_BARS_ROOT).is_dir()


@pytest.mark.skipif(not _corpus_available(), reason="owned ES bars not present")
def test_the_owned_corpus_matches_the_frozen_index() -> None:
    sessions = loader.load_sessions()
    features = loader.session_features(sessions)
    assert len(sessions) == family.M1_ELIGIBLE_SESSIONS == 254
    assert loader.eligible_sessions(features, "M1").sum() == 254
    assert loader.eligible_sessions(features, "M3").sum() == family.GAP_ELIGIBLE_SESSIONS

    rolls = features.loc[features["is_roll_boundary"], "session"].tolist()
    assert tuple(rolls) == family.ROLL_BOUNDARY_SESSIONS


@pytest.mark.skipif(not _corpus_available(), reason="owned ES bars not present")
def test_every_member_replays_the_full_declared_index_on_real_bars() -> None:
    """Structural completeness only: row counts and occupancy bounds. No member
    economics are computed or inspected -- the surrogate gate has not run."""

    features = loader.session_features(loader.load_sessions())
    for member in family.FAMILY:
        result = replay.replay_member(features, member)
        assert len(result) == member.eligible_sessions
        assert result["trades"].isin((0, 1)).all()
        assert result["trades"].sum() <= len(result)
        assert result["side"].isin((-1.0, 0.0, 1.0)).all()
        assert np.isfinite(result["net_points"].to_numpy()).all()
        assert result["fold"].nunique() == 5
