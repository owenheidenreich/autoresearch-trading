"""Resumable driver that normalizes an acquired backfill into builder schema.

**Why this exists rather than an edit.** The normalization *semantics* live in
`normalize_lifecycle_quote_backfill.normalize_session`, which is hash-pinned by
`PREACQUISITION_SEMANTIC_FREEZE_V1.json`; that freeze's post-contact rule forbids
changing any listed source after the preflight and directs development iterations
through the alpha ledger instead of edits to the baseline. Two things nonetheless
have to change around it, and neither is a semantic change:

1. **The pinned `run()` refuses the V5 receipt.** It requires
   `completion.expected_files`, a key the current acquisition runner no longer
   writes, so it raises `acquisition receipt is incomplete` on a receipt that is
   in fact complete (794 sessions, 1,588 files).
2. **It is not resumable.** It refuses to overwrite an existing output by raising,
   so re-running after an interruption marks every already-normalized session
   `DEGRADED`. This job has lost two multi-hour processes to external death
   already; a ~6-8 hour non-resumable pass is a hazard, not a plan.

So this module re-uses the frozen `normalize_session` **unchanged** and replaces
only the orchestration around it. It additionally **verifies the frozen hash
before doing any work**, so the semantics actually applied are provably the ones
the freeze pins — a guarantee the original orchestration never offered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.verify_backfill_clock import inspect_session
from v5.ops.normalize_lifecycle_quote_backfill import (
    ET,
    QuoteNormalizationError,
    canonical_json,
    file_sha256,
    normalize_session,
)

SCHEMA_VERSION = "v5.lifecycle-corpus-normalization.v1"
FREEZE_PATH = Path("v5/work/lifecycle-training/PREACQUISITION_SEMANTIC_FREEZE_V1.json")
PINNED_SOURCE = "v5/ops/normalize_lifecycle_quote_backfill.py"


def assert_frozen_semantics(freeze_path: Path = FREEZE_PATH) -> str:
    """Refuse to run unless the normalizer matches its frozen hash.

    Fails closed: the point of the freeze is that the bytes which shaped the
    corpus are the bytes that were declared, and that is worth proving at the
    moment of use rather than assuming.
    """

    freeze = json.loads(freeze_path.read_text())
    expected = freeze.get("source_files", {}).get(PINNED_SOURCE)
    if not expected:
        raise QuoteNormalizationError(f"freeze does not pin {PINNED_SOURCE}")
    actual = file_sha256(Path(PINNED_SOURCE))
    if actual != expected:
        raise QuoteNormalizationError(
            f"frozen normalizer drifted: expected {expected}, found {actual}"
        )
    return actual


def assert_source_clock(raw_root: Path, *, sample: int = 12) -> dict[str, Any]:
    """Refuse a source root whose delivered sessions lack the terminal bar.

    Two acquisitions of the same 794 sessions exist on disk and differ by exactly
    one bar per session: the superseded V4 corpus ends 15:59 (389 bars, the
    half-open-window defect) and the V5 corpus ends 16:00. They are the same size
    and the same shape, so anything that merely globs a directory cannot tell
    them apart -- and the missing minute is precisely the one terminal accounting
    reads. Naming the old root is a courtesy to a reader; this is the guard that
    makes building from it impossible.

    Tolerant by design: genuine early closes legitimately lack 16:00 (measured, 6
    of 794), so the test is a *majority* over a spread sample rather than a
    unanimity. The separation is not marginal -- the fixed corpus carries the
    terminal minute on ~99% of sessions and the superseded one on 0%.
    """

    files = sorted((raw_root / "raw/databento/opra_spxw_cbbo_1m").glob("*.parquet"))
    if not files:
        raise QuoteNormalizationError(f"no delivered quote files under {raw_root}")
    step = max(1, len(files) // sample)
    chosen = files[::step][:sample]
    with_terminal = 0
    for path in chosen:
        row = inspect_session(path, check_liveness=False)
        with_terminal += int(bool(row["has_terminal_minute"]))
    share = with_terminal / len(chosen)
    if share < 0.6:
        raise QuoteNormalizationError(
            f"source root looks like the superseded 389-bar acquisition: only "
            f"{with_terminal}/{len(chosen)} sampled sessions carry the terminal "
            f"minute. Refusing to normalize from {raw_root}"
        )
    return {
        "sampled_sessions": len(chosen),
        "with_terminal_minute": with_terminal,
        "terminal_minute_share": share,
    }


def acquired_sessions(receipt: dict[str, Any]) -> list[str]:
    """Sessions named by an acquisition receipt, tolerant of receipt shape."""

    rows = receipt.get("files") or []
    sessions = {str(row["session"]) for row in rows if row.get("session")}
    if not sessions:
        raise QuoteNormalizationError("acquisition receipt names no sessions")
    return sorted(sessions)


def _parity_minutes(normalized: pd.DataFrame) -> int:
    stamps = pd.to_datetime(normalized["event_time"], utc=True).dt.tz_convert(ET)
    minutes = stamps.dt.strftime("%H:%M").where(normalized["underlying_price"].notna())
    return int(minutes.nunique())


def normalize_one(
    session: str, *, raw_root: Path, output_root: Path
) -> dict[str, Any]:
    """Normalize a single session through the frozen semantics."""

    definition_path = (
        raw_root / "raw/databento/opra_spxw_definition" / f"{session}.definition.parquet"
    )
    cbbo_path = raw_root / "raw/databento/opra_spxw_cbbo_1m" / f"{session}.cbbo-1m.parquet"
    output = output_root / f"databento_spxw_0dte_{session}.parquet"

    normalized = normalize_session(
        pd.read_parquet(definition_path), pd.read_parquet(cbbo_path), session=session
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temporary path so an interrupted process cannot leave a
    # half-written parquet that a resumed pass would treat as complete.
    staging = output.with_suffix(".partial")
    normalized.to_parquet(staging, index=False)
    staging.replace(output)
    return {
        "session": session,
        "classification": "NORMALIZED",
        "rows": int(len(normalized)),
        "rth_minutes_with_parity_spot": _parity_minutes(normalized),
        "path": str(output),
        "sha256": file_sha256(output),
    }


def run(
    *,
    raw_root: Path,
    output_root: Path,
    acquisition_receipt: Path,
    receipt_path: Path,
    limit: int | None = None,
    freeze_path: Path = FREEZE_PATH,
) -> dict[str, Any]:
    """Normalize every acquired session, skipping work already on disk."""

    frozen_sha = assert_frozen_semantics(freeze_path)
    source_clock = assert_source_clock(raw_root)
    receipt = json.loads(acquisition_receipt.read_text())
    sessions = acquired_sessions(receipt)
    if limit is not None:
        sessions = sessions[:limit]

    results: list[dict[str, Any]] = []
    for session in sessions:
        output = output_root / f"databento_spxw_0dte_{session}.parquet"
        if output.exists():
            results.append(
                {
                    "session": session,
                    "classification": "ALREADY_PRESENT",
                    "path": str(output),
                    "sha256": file_sha256(output),
                }
            )
            continue
        try:
            results.append(
                normalize_one(session, raw_root=raw_root, output_root=output_root)
            )
        except Exception as exc:  # structural failure is data, not a crash
            results.append(
                {
                    "session": session,
                    "classification": "DEGRADED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    def count(name: str) -> int:
        return int(sum(row["classification"] == name for row in results))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "normalize the acquired backfill through frozen semantics; the pinned "
            "run() refuses the current receipt shape and is not resumable"
        ),
        "acquisition_receipt": str(acquisition_receipt),
        "acquisition_receipt_sha256": file_sha256(acquisition_receipt),
        "frozen_normalizer": {"path": PINNED_SOURCE, "sha256": frozen_sha},
        "source_clock_guard": source_clock,
        "driver_sha256": file_sha256(Path(__file__)),
        "source_raw_root": str(raw_root),
        "output_root": str(output_root),
        "sessions": results,
        "summary": {
            "acquired": len(sessions),
            "normalized": count("NORMALIZED"),
            "already_present": count("ALREADY_PRESENT"),
            "degraded": count("DEGRADED"),
        },
    }
    payload["gate"] = "PASS" if count("DEGRADED") == 0 else "DEGRADED_SESSIONS_PRESENT"
    unsigned = dict(payload)
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = run(
        raw_root=args.raw_root,
        output_root=args.output_root,
        acquisition_receipt=args.acquisition_receipt,
        receipt_path=args.receipt,
        limit=args.limit,
    )
    summary = payload["summary"]
    print(
        f"normalization {payload['gate']}: {summary['normalized']} normalized, "
        f"{summary['already_present']} already present, {summary['degraded']} degraded "
        f"of {summary['acquired']} acquired"
    )
    for row in payload["sessions"]:
        if row["classification"] == "DEGRADED":
            print(f"  DEGRADED {row['session']}: {row['reason']}")
    return 0 if payload["gate"] == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
