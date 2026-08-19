"""Assert the delivered quote clock matches the clock the dataset law requires.

**Why this exists.** The V4 acquisition passed every control it had — exact
preflight, ceiling enforced twice, declaration and runner hashes pinned, the
same-day expiry rule applied at request time — and still delivered a corpus that
could not build. Its `cbbo-1m` window ended at 16:00, vendor ranges are half-open,
and CBBO-1m bars are stamped at their **end**, so the bar stamped 16:00:00 was
never delivered: 389 bars per contract instead of 390. That single missing minute
is the one `attach_candidate_outcomes` reads for terminal accounting, and it made
`included_for_episode_build` false for all 794 sessions.

Every existing gate asked whether the request was authorized and affordable.
**None asked whether the bytes that came back carry the clock the builder needs.**
This closes that gap, and is meant to run after any acquisition, before any build.

**Why it also checks liveness.** The V5 re-request fixed the terminal bar, and the
residue exposed a worse defect of the opposite shape. `2022-11-25` — the Friday
after Thanksgiving, an early close — delivers a *full* 390-minute clock ending
16:00 and therefore passes a pure presence check, but its book is **frozen from
13:00 onward**: measured, an identical 237 two-sided contracts every minute to the
close, while every normal session drifts continuously (2025-07-31 514->496,
2022-06-01 191->182). The vendor padded post-close minutes with the last book in
2022 and truncated them from 2023. A presence-only gate admits that session and
hands the builder ~3 hours of fabricated flat prices, where a first-touch label can
never touch and terminal accounting reads a 13:00 quote as a 16:00 settlement.

So presence is not enough: this derives each session's **last live minute** from the
data. One measurement separates all three populations without a hardcoded holiday
calendar — which the vendor's own inconsistency across years shows would be wrong
anyway:

* a normal session stays live to 16:00;
* a truncated early close simply stops (2023-11-24 ends 13:14);
* a padded early close keeps emitting and stops *changing* (2022-11-25).

It reads `ts_recv` always and the top-of-book columns when present, so a
presence-only pass stays cheap and a liveness pass costs one extra column read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from v5.ops.audit_causal_day_coverage import (
    FIRST_QUOTE_MINUTE,
    LAST_QUOTE_MINUTE,
    QUOTE_MINUTES,
)
from v5.ops.download_spxw_history import NY, canonical_json

SCHEMA_VERSION = "v5.backfill-clock-verification.v2"

# Top-of-book columns whose per-minute aggregate forms the liveness signature.
# Absent (as in a presence-only fixture), liveness is reported UNKNOWN rather
# than assumed live -- missing evidence is never a pass.
QUOTE_COLUMNS = ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00")

CLASS_OK = "OK"
CLASS_EARLY_CLOSE = "EARLY_CLOSE_TRUNCATED"
CLASS_STALE_PADDED = "STALE_PADDED"
CLASS_INTERIOR_GAP = "INTERIOR_GAP"


class ClockVerificationError(RuntimeError):
    """The delivered clock cannot support the declared dataset law."""


def session_minutes(path: Path) -> list[str]:
    """Distinct ET minute stamps in one delivered session file."""

    table = pq.read_table(path, columns=["ts_recv"])
    stamps = pd.to_datetime(table.column("ts_recv").to_pandas(), utc=True)
    return sorted(set(stamps.dt.tz_convert(NY).dt.strftime("%H:%M")))


def _available_quote_columns(path: Path) -> tuple[str, ...]:
    present = set(pq.ParquetFile(path).schema_arrow.names)
    return tuple(column for column in QUOTE_COLUMNS if column in present)


def last_live_minute(path: Path) -> tuple[str | None, int]:
    """The last minute whose book differs from the previous minute's.

    Returns ``(minute, trailing_stale_minutes)``. A frozen book repeats its
    entire top-of-book across every contract, so an aggregate signature per
    minute is sufficient and vectorises: across hundreds of contracts the
    chance of a live minute reproducing its predecessor's exact sums is nil.
    ``(None, 0)`` means liveness could not be established from this file.
    """

    columns = _available_quote_columns(path)
    if not columns:
        return None, 0
    # Build from arrow columns explicitly: the owned era stores `ts_recv` as the
    # pandas index, so `to_pandas()` promotes it out of the columns and a
    # frame["ts_recv"] lookup raises on that era alone.
    table = pq.read_table(path, columns=["ts_recv", *columns])
    stamps = pd.to_datetime(table.column("ts_recv").to_pandas(), utc=True).dt.tz_convert(NY)
    frame = pd.DataFrame(
        {name: table.column(name).to_pandas().to_numpy() for name in columns}
    )
    frame["_minute"] = stamps.dt.strftime("%H:%M").to_numpy()
    # Only the minutes the builder actually reads. The owned era was acquired on
    # a wider 08:01-16:01 grid, so walking back from the last *delivered* minute
    # would read a post-close repeat as a frozen book and refuse every session.
    frame = frame[frame["_minute"].isin(set(QUOTE_MINUTES))]
    if frame.empty:
        return None, 0
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    signature = numeric.assign(_minute=frame["_minute"]).groupby("_minute").agg(
        ["size", "sum"]
    )
    ordered = signature.sort_index()
    if len(ordered) < 2:
        return (ordered.index[-1] if len(ordered) else None), 0
    rows = ordered.to_numpy(dtype=float)
    stale = 0
    for index in range(len(rows) - 1, 0, -1):
        if not (rows[index] == rows[index - 1]).all():
            break
        stale += 1
    return str(ordered.index[len(ordered) - 1 - stale]), stale


def inspect_session(path: Path, *, check_liveness: bool = True) -> dict[str, Any]:
    """Compare one session's delivered minutes against the required clock."""

    minutes = session_minutes(path)
    present = set(minutes)
    required = set(QUOTE_MINUTES)
    missing = sorted(required - present)
    complete = not missing

    live_minute, stale_tail = (None, 0)
    if check_liveness:
        live_minute, stale_tail = last_live_minute(path)

    # Classify within the required window only. The owned era is delivered on a
    # wider 08:01-16:01 grid, so anchoring contiguity on the delivered first
    # minute mislabels its genuine early closes as interior gaps.
    in_window = sorted(present & required)
    last_required = in_window[-1] if in_window else None
    contiguous_to_stop = bool(in_window) and in_window[0] == FIRST_QUOTE_MINUTE and not [
        m for m in QUOTE_MINUTES if m <= last_required and m not in present
    ]

    if live_minute is not None and stale_tail:
        classification = CLASS_STALE_PADDED
    elif complete:
        classification = CLASS_OK
    elif contiguous_to_stop:
        # Contiguous from the open to an early stop: the session really ended.
        classification = CLASS_EARLY_CLOSE
    else:
        classification = CLASS_INTERIOR_GAP

    return {
        "session": path.name.split(".")[0],
        "delivered_minutes": len(minutes),
        "required_minutes": len(QUOTE_MINUTES),
        "first_minute": minutes[0] if minutes else None,
        "last_minute": minutes[-1] if minutes else None,
        "has_terminal_minute": LAST_QUOTE_MINUTE in present,
        "missing_required_minutes": len(missing),
        "missing_examples": missing[:5],
        "last_live_minute": live_minute,
        "stale_trailing_minutes": stale_tail,
        "liveness_checked": live_minute is not None,
        "classification": classification,
        # Build eligibility is stricter than presence: a padded session has the
        # full clock and must still be refused.
        "build_eligible": complete and not stale_tail,
        "ok": complete and not stale_tail,
    }


def verify(
    root: Path, *, limit: int | None = None, check_liveness: bool = True
) -> dict[str, Any]:
    """Verify every delivered session; fail closed on the first shortfall class."""

    files = sorted(root.glob("*.parquet"))
    if not files:
        raise ClockVerificationError(f"no delivered quote files under {root}")
    if limit is not None:
        files = files[:limit]
    rows = [inspect_session(path, check_liveness=check_liveness) for path in files]
    failed = [row for row in rows if not row["ok"]]
    without_terminal = [row for row in rows if not row["has_terminal_minute"]]
    by_class: dict[str, list[str]] = {}
    for row in rows:
        by_class.setdefault(row["classification"], []).append(row["session"])
    stale = by_class.get(CLASS_STALE_PADDED, [])
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "assert the delivered quote clock supports the dataset law; the V4 window "
            "silently dropped the terminal 16:00 bar and no gate noticed, and a padded "
            "early close carries a full clock whose final hours are a frozen book"
        ),
        "root": str(root),
        "required_clock": {
            "first": FIRST_QUOTE_MINUTE,
            "last": LAST_QUOTE_MINUTE,
            "minutes": len(QUOTE_MINUTES),
        },
        "sessions": len(rows),
        "sessions_ok": len(rows) - len(failed),
        "sessions_failed": len(failed),
        "sessions_without_terminal_minute": len(without_terminal),
        "liveness_checked": all(row["liveness_checked"] for row in rows),
        "classification_counts": {name: len(v) for name, v in sorted(by_class.items())},
        # The artifact the corpus build consumes: presence alone would admit a
        # padded session, so eligibility is published rather than re-derived.
        "build_eligible_sessions": sorted(
            row["session"] for row in rows if row["build_eligible"]
        ),
        "excluded_sessions": {
            name: sorted(sessions)
            for name, sessions in sorted(by_class.items())
            if name != CLASS_OK
        },
        "gate": "PASS" if not failed else "FAIL_CLOCK_SHORTFALL",
        "failures": failed[:20],
    }
    if stale:
        receipt["stale_padded_warning"] = (
            "these sessions report a complete clock while their book is frozen; "
            "presence-only checks admit them and the builder would read fabricated "
            "flat prices as real quotes"
        )
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-liveness",
        action="store_true",
        help="presence-only pass; cheaper, but cannot see a padded early close",
    )
    args = parser.parse_args()

    receipt = verify(args.root, limit=args.limit, check_liveness=not args.no_liveness)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(
        f"clock verification {receipt['gate']}: "
        f"{receipt['sessions_ok']}/{receipt['sessions']} sessions carry the full "
        f"{receipt['required_clock']['minutes']}-minute clock; "
        f"{receipt['sessions_without_terminal_minute']} lack {LAST_QUOTE_MINUTE}"
    )
    for name, sessions in sorted(receipt.get("excluded_sessions", {}).items()):
        print(f"  {name}: {len(sessions)} -> {', '.join(sessions[:6])}")
    print(f"  build-eligible: {len(receipt['build_eligible_sessions'])}")
    return 0 if receipt["gate"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
