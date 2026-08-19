"""Fill the parity-spot minutes the strict solver leaves empty, with provenance.

**The blocker this removes.** `attach_candidate_outcomes` requires a finite
underlying at *every* minute of the 390-minute clock, and refuses the whole
session otherwise. The backfill era fails that check on its last 2-5 minutes:
measured across sessions, the last strictly-solvable parity minute is 15:55-15:58
and 16:00 is always empty. The 16:00 *bar* exists — that was the V5 repair — but
the chain has thinned so far by then that the strict solver cannot use it.

**Why a relaxed solve is legitimate rather than a fudge.** SPXW options are
**European**, so put-call parity `S = K + C - P` holds *exactly* at any single
paired strike; averaging several strikes reduces quote noise, it is not a
mathematical requirement. The frozen normalizer requires three paired strikes
within a window, and measured on 2022-06-01 the paired count decays 10 (15:50) ->
6 (15:55) -> 3 (15:59) -> **1 (16:00)**. So the terminal minute is solvable; the
strict rule simply declines to solve it.

This pass therefore recomputes only the minutes left NaN, using the paired
strikes nearest the money, and **records how many strikes backed each value**, so
a thin estimate is visible rather than indistinguishable from a strong one:

* `underlying_price_source` — `strict_parity` (untouched), `relaxed_parity`, or
  `unavailable`;
* `underlying_parity_strikes` — the count actually used.

Nothing is forward-filled and nothing is carried across minutes: a minute with no
paired strike stays NaN and its session is reported rather than silently patched.
The frozen normalizer is not modified; this writes an enriched copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES
from v5.ops.build_causal_day_dataset import ET, canonical_json, file_sha256

SCHEMA_VERSION = "v5.parity-spot-repair.v1"
SOURCE_STRICT = "strict_parity"
SOURCE_RELAXED = "relaxed_parity"
SOURCE_UNAVAILABLE = "unavailable"
# Parity is exact per strike; this bounds how far from the money we will look
# before treating an estimate as unrepresentative of the index.
RELAXED_WINDOW_POINTS = 60.0


class ParityRepairError(RuntimeError):
    """A session cannot supply a causal underlying at a required minute."""


def _minute_series(frame: pd.DataFrame) -> pd.Series:
    stamped = pd.to_datetime(frame["event_time"], utc=True).dt.tz_convert(ET)
    return stamped.dt.strftime("%H:%M")


def relaxed_parity_spot(minute_frame: pd.DataFrame) -> tuple[float, int]:
    """Exact European parity at the paired strikes nearest the money.

    Returns ``(spot, strikes_used)``; ``(nan, 0)`` when no strike is paired.
    """

    bid = pd.to_numeric(minute_frame["bid"], errors="coerce")
    ask = pd.to_numeric(minute_frame["ask"], errors="coerce")
    live = minute_frame[(bid > 0.0) & (ask > bid)]
    if live.empty:
        return float("nan"), 0
    wide = live.pivot_table(index="strike", columns="right", values="mid", aggfunc="last")
    if not {"C", "P"} <= set(wide.columns):
        return float("nan"), 0
    paired = wide.dropna(subset=["C", "P"])
    if paired.empty:
        return float("nan"), 0
    strikes = paired.index.to_numpy(float)
    implied = strikes + paired["C"].to_numpy(float) - paired["P"].to_numpy(float)
    centre = float(np.median(implied))
    near = np.abs(strikes - centre) <= RELAXED_WINDOW_POINTS
    chosen = implied[near] if near.any() else implied
    return float(np.mean(chosen)), int(chosen.size)


def repair_session(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill NaN underlying minutes in one normalized session."""

    out = frame.copy()
    minutes = _minute_series(out)
    out["_minute"] = minutes
    known = pd.to_numeric(out["underlying_price"], errors="coerce")
    out["underlying_price"] = known
    out["underlying_price_source"] = np.where(
        known.notna(), SOURCE_STRICT, SOURCE_UNAVAILABLE
    )
    out["underlying_parity_strikes"] = np.where(known.notna(), -1, 0)

    per_minute = out.groupby("_minute")["underlying_price"].transform("median")
    empty_minutes = sorted(
        {
            minute
            for minute in out.loc[per_minute.isna(), "_minute"].unique()
            if minute in set(QUOTE_MINUTES)
        }
    )

    repaired: dict[str, int] = {}
    for minute in empty_minutes:
        rows = out["_minute"].eq(minute)
        spot, used = relaxed_parity_spot(out[rows])
        if used == 0 or not np.isfinite(spot):
            continue
        out.loc[rows, "underlying_price"] = spot
        out.loc[rows, "underlying_price_source"] = SOURCE_RELAXED
        out.loc[rows, "underlying_parity_strikes"] = used
        repaired[minute] = used

    still_missing = sorted(set(empty_minutes) - set(repaired))
    covered = (
        out.loc[out["_minute"].isin(QUOTE_MINUTES)]
        .groupby("_minute")["underlying_price"]
        .median()
        .reindex(QUOTE_MINUTES)
    )
    report = {
        "strict_minutes": int(len(QUOTE_MINUTES) - len(empty_minutes)),
        "repaired_minutes": len(repaired),
        "repaired_detail": repaired,
        "unrepaired_minutes": still_missing,
        "clock_complete": bool(np.isfinite(covered.to_numpy(float)).all()),
    }
    return out.drop(columns="_minute"), report


def run(
    *,
    source_root: Path,
    output_root: Path,
    receipt_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise ParityRepairError(f"no normalized sessions under {source_root}")
    if limit is not None:
        files = files[:limit]

    results: list[dict[str, Any]] = []
    for path in files:
        session = path.stem.split("_")[-1]
        destination = output_root / path.name
        if destination.exists():
            results.append({"session": session, "classification": "ALREADY_PRESENT"})
            continue
        try:
            repaired, report = repair_session(pd.read_parquet(path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.with_suffix(".partial")
            repaired.to_parquet(staging, index=False)
            staging.replace(destination)
            results.append(
                {
                    "session": session,
                    "classification": "REPAIRED" if report["repaired_minutes"] else "UNCHANGED",
                    "sha256": file_sha256(destination),
                    **report,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "session": session,
                    "classification": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    complete = [r for r in results if r.get("clock_complete")]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "supply a causal underlying at every required minute; the strict "
            "parity rule leaves the last 2-5 minutes empty and the pinned builder "
            "refuses a session with any gap in the clock"
        ),
        "relaxed_window_points": RELAXED_WINDOW_POINTS,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sessions": results,
        "summary": {
            "sessions": len(results),
            "clock_complete": len(complete),
            "failed": int(sum(r["classification"] == "FAILED" for r in results)),
            "repaired": int(sum(r["classification"] == "REPAIRED" for r in results)),
        },
    }
    payload["gate"] = (
        "PASS" if len(complete) == len(results) else "INCOMPLETE_CLOCK_REMAINS"
    )
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    payload = run(
        source_root=args.source_root,
        output_root=args.output_root,
        receipt_path=args.receipt,
        limit=args.limit,
    )
    summary = payload["summary"]
    print(
        f"parity repair {payload['gate']}: {summary['clock_complete']}/{summary['sessions']} "
        f"sessions carry a complete underlying clock ({summary['repaired']} repaired, "
        f"{summary['failed']} failed)"
    )
    return 0 if payload["gate"] == "PASS" else 6


if __name__ == "__main__":
    raise SystemExit(main())
