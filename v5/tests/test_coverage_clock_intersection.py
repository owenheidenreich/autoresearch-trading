"""Coverage eligibility must AND in the delivered-clock liveness verdict.

The defect this guards: a padded early close satisfies every presence check the
coverage audit makes (`missing_rth_quote_minutes == 0`) while its final hours are
a frozen book. 2022-11-25 is the measured instance -- an identical top-of-book
from 13:00 to 16:00 under a complete 390-minute clock.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v5.ops.audit_causal_day_coverage import CoverageError, clock_eligible_sessions


def _receipt(tmp_path: Path, name: str, eligible: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"build_eligible_sessions": eligible}))
    return path


def test_sessions_from_several_roots_are_unioned(tmp_path: Path) -> None:
    """Two eras, two receipts, one eligibility set."""

    backfill = _receipt(tmp_path, "backfill.json", ["2022-06-01", "2023-01-03"])
    owned = _receipt(tmp_path, "owned.json", ["2025-08-01"])
    assert clock_eligible_sessions([backfill, owned]) == {
        "2022-06-01",
        "2023-01-03",
        "2025-08-01",
    }


def test_a_v1_receipt_without_the_field_is_refused(tmp_path: Path) -> None:
    """A presence-only receipt must not silently pass as a liveness verdict."""

    stale = tmp_path / "v1.json"
    stale.write_text(json.dumps({"gate": "PASS", "sessions_ok": 794}))
    with pytest.raises(CoverageError, match="needs v2"):
        clock_eligible_sessions([stale])


def test_receipts_certifying_nothing_are_refused(tmp_path: Path) -> None:
    empty = _receipt(tmp_path, "empty.json", [])
    with pytest.raises(CoverageError, match="no eligible session"):
        clock_eligible_sessions([empty])


def test_a_padded_session_is_removed_by_the_intersection() -> None:
    """The 2022-11-25 shape: presence-eligible, liveness-ineligible."""

    coverage = pd.DataFrame(
        {
            "session": ["2022-06-01", "2022-11-25"],
            # Both pass every presence-based condition.
            "included_for_episode_build": [True, True],
        }
    )
    certified = {"2022-06-01"}

    clock_ok = coverage["session"].astype(str).isin(certified)
    excluded = sorted(
        coverage.loc[coverage["included_for_episode_build"] & ~clock_ok, "session"]
    )
    coverage["included_for_episode_build"] &= clock_ok

    assert excluded == ["2022-11-25"]
    assert coverage["included_for_episode_build"].tolist() == [True, False]
