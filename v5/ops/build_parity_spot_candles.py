"""Derive the session tape from SPX parity spot instead of ES futures.

**Owner ruling, 2026-08-19.** The active policy is SPXW/SPX-only with no futures
input, so the chart state the model reads must be SPX-derived. The corpus built
on 2026-08-18 used ES 1-minute bars; the pre-fit review measured the basis at
+20.3/+29.8 points and showed it cancels in the four difference-based tape
channels the declared member actually reads. The ruling is nonetheless to
rebuild, and the reason is provenance rather than arithmetic: "the candles are
ES" is the kind of inconsistency a later session discovers mid-fit.

**What this produces.** One ES-shaped parquet per session, so the pinned
`build_causal_day_dataset.build_session` consumes it unchanged -- the freeze's
post-contact rule forbids editing that file, and nothing here needs to.
Everything downstream follows automatically: only the `candles` and `minutes`
tables depend on this input. The ladder, the candidates, every label and the
atlas are computed from the quote file's own `underlying_price`, so they were
already SPX-denominated and are byte-identical across the rebuild.

**The clock, and why the offset is what it is.** An ES bar stamped `t` is
knowable at `t+1`, which is how the builder keeps the tape causal. A `cbbo-1m`
quote snapshot stamped `t` *is* the market at `t`. So the bar labelled `t` is
filled from the quote snapshot at `t+1`: bar 09:30 from the 09:31 snapshot,
through bar 15:59 from the 16:00 snapshot. That is exactly 390 bars from exactly
390 quote minutes, and every bar is still first readable one minute after the
state it describes.

**Open, high and low equal the close, and that is a statement rather than a
defect.** A one-minute CBBO snapshot is a single observation of the index; there
is no intra-minute range to report and inventing one would be fabrication. The
four channels the member reads -- `close_from_session_open_points`,
`range_position`, `return_1m`, `realised_vol_15m` -- are all functions of the
close series, and `range_position` is computed from running extremes of that
series, which is well defined. Candle anatomy (body, wicks) is therefore
identically zero here, and it is already barred from the member's state as
census-dead one-minute noise.

**Volume is zero because the index has no volume.** The design bars every volume
channel for exactly this reason. `log_volume` becomes 0 and
`volume_vs_expanding_median` becomes NaN in the `minutes` table; the tensorizer's
own guard maps the latter to 0.0, so the observation tensor stays finite.

Nothing here contacts a vendor, spends money, or fits anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES
from v5.ops.build_causal_day_dataset import (
    QUOTE_COLUMNS,
    canonical_json,
    minute_label,
    minute_number,
    prepare_quotes,
)

SCHEMA_VERSION = "v5.parity-spot-candles.v1"
TAPE_SOURCE = "spx_parity_spot"
SESSION_FILE = re.compile(r"^databento_spxw_0dte_(20\d{2}-\d{2}-\d{2})\.parquet$")

#: The ES grid the pinned `prepare_es` insists on: 09:30 through 15:59.
FIRST_BAR_MINUTE = "09:30"
LAST_BAR_MINUTE = "15:59"
BAR_MINUTES = tuple(
    minute_label(value)
    for value in range(minute_number(FIRST_BAR_MINUTE), minute_number(LAST_BAR_MINUTE) + 1)
)
ET = "America/New_York"


class ParitySpotCandleError(RuntimeError):
    """A session cannot produce the declared SPX tape."""


def parity_spot_by_minute(quote_path: Path, session: str) -> pd.Series:
    """Median parity spot per quote minute, read through the pinned normalizer.

    Reading through `prepare_quotes` means this is the same underlying series the
    builder's labels and the entry band already use, not a separately derived
    number that could drift from them.
    """

    raw = pd.read_parquet(quote_path, columns=list(QUOTE_COLUMNS))
    quotes = prepare_quotes(raw, session)
    return (
        quotes.groupby("minute")["underlying_price"]
        .median()
        .reindex(QUOTE_MINUTES)
        .astype(float)
    )


def parity_spot_candles(spot: pd.Series, session: str) -> pd.DataFrame:
    """Bar `t` from the snapshot at `t+1`; open/high/low all equal the close."""

    observed = [minute_label(minute_number(bar) + 1) for bar in BAR_MINUTES]
    missing = [
        bar
        for bar, source in zip(BAR_MINUTES, observed, strict=True)
        if source not in spot.index or not np.isfinite(spot.get(source, np.nan))
    ]
    if missing:
        raise ParitySpotCandleError(
            f"{session}: parity spot absent at {len(missing)} of {len(BAR_MINUTES)} bars "
            f"(first {missing[0]}); repair the quote file before building the tape"
        )
    close = np.asarray([float(spot[source]) for source in observed], dtype=float)
    if (close <= 0.0).any():
        raise ParitySpotCandleError(f"{session}: non-positive parity spot in the session grid")

    stamps = pd.to_datetime(
        [f"{session} {bar}" for bar in BAR_MINUTES]
    ).tz_localize(ET).tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.zeros(len(close), dtype=float),
            # Provenance the pinned `prepare_es` ignores and a human reader does
            # not: which snapshot filled this bar, and what the tape is made of.
            "bar_observation_minute": observed,
            "tape_source": TAPE_SOURCE,
        },
        index=pd.Index(stamps, name="ts_event"),
    )
    return frame


def build_one(session: str, *, quote_path: Path, out_dir: Path) -> dict[str, Any]:
    spot = parity_spot_by_minute(quote_path, session)
    candles = parity_spot_candles(spot, session)
    destination = out_dir / f"{session}.es_c_0.ohlcv-1m.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".partial")
    candles.to_parquet(staging)
    staging.replace(destination)

    close = candles["close"].to_numpy(float)
    steps = np.abs(np.diff(close))
    return {
        "session": session,
        "classification": "BUILT",
        "bars": int(len(candles)),
        "open_spx": float(close[0]),
        "close_spx": float(close[-1]),
        "session_range_points": float(close.max() - close.min()),
        # A QC signal, reported and never gated: a parity solve that broke would
        # show up as an implausible one-minute step long before it showed up in
        # a label.
        "max_one_minute_step_points": float(steps.max()) if steps.size else 0.0,
    }


def locate(quote_roots: Sequence[Path], session: str) -> Path:
    for root in quote_roots:
        candidate = Path(root) / f"databento_spxw_0dte_{session}.parquet"
        if candidate.exists():
            return candidate
    raise ParitySpotCandleError(f"{session}: no quote file in any declared root")


def sessions_in(roots: Sequence[Path]) -> list[str]:
    found: set[str] = set()
    for root in roots:
        for path in Path(root).glob("*.parquet"):
            match = SESSION_FILE.match(path.name)
            if match:
                found.add(match.group(1))
    return sorted(found)


def run(
    *,
    quote_roots: Sequence[Path],
    sessions: Sequence[str],
    out_dir: Path,
    receipt_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(str(session) for session in sessions)
    if not ordered:
        raise ParitySpotCandleError("no session supplied")
    if limit is not None:
        ordered = ordered[:limit]

    results: list[dict[str, Any]] = []
    for session in ordered:
        destination = out_dir / f"{session}.es_c_0.ohlcv-1m.parquet"
        if destination.exists():
            results.append({"session": session, "classification": "ALREADY_PRESENT"})
            continue
        try:
            results.append(
                build_one(session, quote_path=locate(quote_roots, session), out_dir=out_dir)
            )
        except Exception as exc:
            results.append(
                {
                    "session": session,
                    "classification": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    built = [row for row in results if row["classification"] == "BUILT"]
    steps = [row["max_one_minute_step_points"] for row in built]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "owner ruling 2026-08-19: the tape is SPX-derived (parity spot), not "
            "ES futures; emitted in the ES file shape so the pinned build_session "
            "consumes it unedited"
        ),
        "tape_source": TAPE_SOURCE,
        "clock": {
            "bar_grid": f"{FIRST_BAR_MINUTE}-{LAST_BAR_MINUTE}",
            "bars_per_session": len(BAR_MINUTES),
            "bar_filled_from": "the quote snapshot one minute later",
            "ohlc": "open=high=low=close; a 1-minute CBBO snapshot has no intra-minute range",
            "volume": "0.0; the index has no volume and every volume channel is barred",
        },
        "quote_roots": [str(root) for root in quote_roots],
        "out_dir": str(out_dir),
        "sessions": results,
        "summary": {
            "requested": len(ordered),
            "built": len(built),
            "already_present": int(
                sum(row["classification"] == "ALREADY_PRESENT" for row in results)
            ),
            "failed": int(sum(row["classification"] == "FAILED" for row in results)),
            "max_one_minute_step_points": float(max(steps)) if steps else None,
            "median_one_minute_step_points": float(np.median(steps)) if steps else None,
        },
    }
    payload["gate"] = "PASS" if payload["summary"]["failed"] == 0 else "SESSIONS_FAILED"
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote-root", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sessions = args.session or sessions_in(args.quote_root)
    payload = run(
        quote_roots=args.quote_root,
        sessions=sessions,
        out_dir=args.out_dir,
        receipt_path=args.receipt,
        limit=args.limit,
    )
    summary = payload["summary"]
    print(
        f"parity-spot tape {payload['gate']}: {summary['built']} built, "
        f"{summary['already_present']} present, {summary['failed']} failed "
        f"of {summary['requested']}"
    )
    print(
        f"  one-minute step: median {summary['median_one_minute_step_points']}, "
        f"max {summary['max_one_minute_step_points']} points"
    )
    for row in payload["sessions"]:
        if row["classification"] == "FAILED":
            print(f"  FAILED {row['session']}: {row['reason']}")
    return 0 if payload["gate"] == "PASS" else 7


if __name__ == "__main__":
    raise SystemExit(main())
