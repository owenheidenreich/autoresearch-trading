from __future__ import annotations

import numpy as np
import pandas as pd

from v4.research.pathd_phase1_entry import (
    ENTRY_FEATURE_NAMES,
    entry_expanding_folds,
    development_sessions,
    select_deterministic_control_policy,
    select_oof_entry_policy,
)


def _candidates() -> pd.DataFrame:
    rows = []
    for minute, momentum in ((31, 5.0), (32, -8.0), (90, -4.0)):
        boundary = pd.Timestamp("2026-08-03 09:30", tz="America/New_York") + pd.Timedelta(
            minutes=minute
        )
        for right in ("C", "P"):
            for offset in (-5.0, 0.0, 5.0):
                rows.append(
                    {
                        "session": "2026-08-03",
                        "feature_boundary_ns": boundary.value,
                        "candidate_uid": f"{minute}-{right}-{offset}",
                        "right": right,
                        "strike_offset": offset,
                        "momentum_15m_bps": momentum,
                        "calibrated_prediction": 10.0 if (minute, right, offset) == (32, "P", 0.0) else 1.0,
                        "outer_fold": 4,
                    }
                )
    return pd.DataFrame(rows)


def test_learned_policy_uses_score_side_then_nearest_atm() -> None:
    selected = select_oof_entry_policy(_candidates())
    assert len(selected) == 2
    assert selected.iloc[0]["candidate_uid"] == "31-C-0.0"
    assert selected.iloc[0]["signal_prediction"] == 1.0


def test_control_ignores_scores_and_uses_completed_momentum() -> None:
    candidates = _candidates()
    candidates["calibrated_prediction"] = np.arange(len(candidates), dtype=float) * 1_000.0
    selected = select_deterministic_control_policy(candidates)
    assert selected["candidate_uid"].tolist() == ["31-C-0.0", "90-P-0.0"]
    assert selected["signal_prediction"].isna().all()


def test_entry_registry_and_folds_are_frozen() -> None:
    assert len(ENTRY_FEATURE_NAMES) == 18
    assert ENTRY_FEATURE_NAMES[-1] == "is_call"
    sessions = pd.bdate_range("2025-01-02", periods=100).strftime("%Y-%m-%d").tolist()
    folds = entry_expanding_folds(sessions)
    assert len(folds) == 5
    tests = []
    for fold in folds:
        assert len(fold["embargo"]) == 1
        assert set(fold["train"]).isdisjoint(fold["test"])
        tests.extend(fold["test"])
    assert len(tests) == len(set(tests))


def test_corpus_inventory_excludes_the_36_session_firewall(tmp_path) -> None:
    directory = tmp_path / "raw/databento/opra_spxw_cbbo_1m"
    directory.mkdir(parents=True)
    sessions = pd.bdate_range("2025-08-01", periods=251).strftime("%Y-%m-%d")
    for session in sessions:
        (directory / f"{session}.cbbo-1m.parquet").touch()
    allowed = development_sessions(tmp_path)
    assert len(allowed) == 215
    assert allowed[-1] == sessions[214]
    assert set(allowed).isdisjoint(sessions[215:])
