from __future__ import annotations

import numpy as np
import pandas as pd

from v4.research.pathd_wave2_signal_discovery import (
    COMPONENTS,
    MEMBERS,
    SCORE_COLUMNS,
    _causal_audit,
    _robust_past_z,
    feature_lineage_manifest,
    session_shuffled_score,
    strict_serial_replay,
)


MINUTE_NS = 60_000_000_000


def test_frozen_family_has_exactly_three_equal_weight_nonfitted_scores() -> None:
    lineage = feature_lineage_manifest()
    assert tuple(lineage["members"]) == MEMBERS
    assert set(COMPONENTS) == set(MEMBERS)
    assert len(MEMBERS) == 3
    for member in MEMBERS:
        row = lineage["members"][member]
        assert row["weight"] == 1.0 / len(COMPONENTS[member])
        assert row["threshold"] == 1.0
    assert lineage["normalization"]["fitted"] is False


def test_robust_normalization_is_past_only_requires_30_minutes_and_separates_rights() -> None:
    rows = []
    for minute in range(40):
        for right, sign in (("C", 1.0), ("P", -1.0)):
            rows.append(
                {
                    "feature_boundary_ns": minute * MINUTE_NS,
                    "right": right,
                    "component": sign * minute,
                }
            )
    frame = pd.DataFrame(rows)
    z = _robust_past_z(frame, "component")
    assert z.iloc[: 30 * 2].isna().all()
    call = z[(frame["right"].eq("C")) & frame["feature_boundary_ns"].eq(30 * MINUTE_NS)]
    put = z[(frame["right"].eq("P")) & frame["feature_boundary_ns"].eq(30 * MINUTE_NS)]
    assert call.iloc[0] > 0.0
    assert put.iloc[0] < 0.0
    mutated = frame.copy()
    mutated.loc[mutated["feature_boundary_ns"] > 30 * MINUTE_NS, "component"] = 1e9
    rerun = _robust_past_z(mutated, "component")
    assert rerun[frame["feature_boundary_ns"].eq(30 * MINUTE_NS)].equals(
        z[frame["feature_boundary_ns"].eq(30 * MINUTE_NS)]
    )


def _replay_frame() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-02 15:00:00", tz="UTC").value
    rows = []
    for minute, score, symbol, pnl in (
        (0, 1.5, "SPXW  260102C06000000", 100.0),
        (0, 2.0, "SPXW  260102P06000000", 50.0),
        (30, 3.0, "SPXW  260102C06005000", 500.0),
    ):
        decision = base + minute * MINUTE_NS
        rows.append(
            {
                "session": "2026-01-02",
                "outer_fold": 0,
                "candidate_uid": symbol,
                "raw_symbol": symbol,
                "feature_boundary_ns": decision,
                "decision_emission_ns": decision,
                "entry_arrival_ns": decision + 1_000_000_000,
                "score": score,
                "relative_spread": 0.05,
                "decision_mid": 5.0,
                "full_horizon_eligible": True,
                "filled_60m": True,
                "pnl_60m": pnl,
                "limit_60m": 5.0,
                "fill_time_ns_60m": decision + 2_000_000_000,
                "exit_time_ns_60m": decision + 2_000_000_000 + 60 * MINUTE_NS,
            }
        )
    return pd.DataFrame(rows)


def test_strict_serial_replay_selects_high_score_and_blocks_overlap() -> None:
    attempts, sessions, diagnostics = strict_serial_replay(
        _replay_frame(), score_column="score", horizon_min=60
    )
    assert attempts["candidate_uid"].tolist() == ["SPXW  260102P06000000"]
    assert attempts["pnl"].tolist() == [50.0]
    assert sessions["net_pnl"].tolist() == [50.0]
    assert diagnostics["filled_trades"] == 1


def test_session_shuffle_moves_scores_between_sessions_inside_exact_stratum() -> None:
    rows = []
    for fold in (0,):
        for session, score in (("2026-01-02", 1.0), ("2026-01-05", 9.0)):
            for minute in range(3):
                rows.append(
                    {
                        "session": session,
                        "outer_fold": fold,
                        "right": "C",
                        "strike_offset": 0.0,
                        "feature_boundary_ns": pd.Timestamp(
                            f"{session} 15:{minute:02d}:00", tz="UTC"
                        ).value,
                        "raw_symbol": f"{session}-{minute}",
                        "score": score,
                    }
                )
    frame = pd.DataFrame(rows)
    shuffled = session_shuffled_score(frame, "score", label="test")
    assert set(shuffled[frame["session"].eq("2026-01-02")]) == {9.0}
    assert set(shuffled[frame["session"].eq("2026-01-05")]) == {1.0}


def test_causal_audit_rejects_post_decision_score_source_and_future_outcomes_do_not_mutate_score() -> None:
    frame = pd.DataFrame(
        {
            "candidate_uid": ["a", "b"],
            "decision_emission_ns": [100, 100],
            "option_available_at_ns": [90, 90],
            "option_volume_available_at_ns": [100, 101],
            "cross_market_available_at_ns": [100, 100],
            "score_surface": [1.0, 1.0],
            "score_microstructure": [1.0, 1.0],
            "score_cross_market": [1.0, 1.0],
            "pnl_30m": [1.0, 2.0],
            "pnl_60m": [3.0, 4.0],
            "fill_time_ns_60m": [200, 200],
            "exit_time_ns_60m": [300, 300],
        }
    )
    audit = _causal_audit(frame)
    assert audit["clock_violations"][MEMBERS[1]] == 1
    assert not audit["causal_clock_passed"]
    assert audit["future_mutation_passed"]
    assert set(SCORE_COLUMNS) == set(MEMBERS)
