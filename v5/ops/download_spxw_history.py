"""Acquire SPXW 0DTE one-minute bars, free under the existing subscription.

Only ``ohlcv-1m`` is requested. The quote schemas are what cost money outside the
trailing twelve months and are forbidden by the manifest; the payoff constants
this corpus feeds were already measured against quotes on the owned year.

Per-session 0DTE membership is **derived, never assumed**: a contract counts as
0DTE only when the expiry encoded in its own OSI symbol equals the session date.
Sessions that turn out to have no same-day expiry are written as empty and are
skipped downstream, the same way the ES equity-holiday rule was derived from
close times rather than taken from a calendar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DATASET = "OPRA.PILLAR"
SCHEMA = "ohlcv-1m"
PARENT = "SPXW.OPT"
NY = ZoneInfo("America/New_York")
# SPXW OSI: root, two spaces, YYMMDD, C/P, 8-digit strike in thousandths.
OSI = re.compile(r"^SPXW\s+(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


def _retry(label: str, call, *, attempts: int = 5):
    """Retry a vendor call through transient network faults.

    A multi-hour acquisition will meet at least one dropped connection, and
    losing 775 completed sessions to one ``RemoteDisconnected`` is a bug in the
    runner rather than a fact about the data. Errors that are *about* the
    request -- an unresolvable symbol on a holiday, an authorisation problem --
    are re-raised immediately so they are not retried into a silent stall.
    """

    permanent = ("symbology_invalid_request", "authentication", "403", "422")
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            text = str(exc)
            if any(token in text for token in permanent) or attempt == attempts:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    retry {attempt}/{attempts - 1} for {label} in {wait}s: {text[:70]}", flush=True)
            time.sleep(wait)


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _sessions(start: date, end: date) -> list[date]:
    out, cursor = [], start
    while cursor <= end:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _bounds(session: date) -> tuple[str, str]:
    from datetime import datetime, timezone

    base = datetime.combine(session, datetime.min.time(), tzinfo=NY)
    start = base.replace(hour=9, minute=30).astimezone(timezone.utc)
    end = base.replace(hour=16).astimezone(timezone.utc)
    fmt = lambda t: t.isoformat().replace("+00:00", "Z")  # noqa: E731
    return fmt(start), fmt(end)


def _zero_dte(frame: pd.DataFrame, session: date) -> pd.DataFrame:
    """Rows whose OSI symbol encodes an expiry equal to the session date."""

    if frame.empty or "symbol" not in frame.columns:
        return frame.iloc[0:0]
    stamp = session.strftime("%y%m%d")
    # The bar index carries one timestamp per contract, so it is full of
    # duplicate labels. Align positionally rather than by label.
    flat = frame.reset_index()
    parsed = flat["symbol"].astype(str).str.extract(OSI)
    keep = parsed["expiry"].eq(stamp).fillna(False).to_numpy()
    if not keep.any():
        return flat.iloc[0:0]
    out = flat.loc[keep].copy()
    out["strike"] = parsed.loc[keep, "strike"].astype(float).to_numpy() / 1000.0
    out["right"] = parsed.loc[keep, "right"].to_numpy()
    return out.reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-06-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--max-cost", type=float, default=5.0)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/spxw_0dte_2022-06-01_2026-07-31"),
    )
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cap = float(manifest["authorized_scope"]["paid_cap_usd"])
    if cap > args.max_cost:
        raise SystemExit(f"abort: manifest cap {cap} exceeds --max-cost {args.max_cost}")

    _load_env(args.env_file)
    import databento as db

    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    out_dir = args.root / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = _sessions(
        date.fromisoformat(args.start), date.fromisoformat(args.end)
    )
    spent, written, empty, records = 0.0, 0, 0, []
    for i, session in enumerate(sessions, 1):
        target = out_dir / f"{session.isoformat()}.spxw_0dte.ohlcv-1m.parquet"
        if target.exists():
            continue
        start, end = _bounds(session)
        try:
            cost = float(
                _retry(
                    f"{session} cost",
                    lambda: client.metadata.get_cost(
                        dataset=DATASET, schema=SCHEMA, symbols=[PARENT],
                        stype_in="parent", start=start, end=end,
                    ),
                )
            )
        except Exception as exc:  # market holiday: the parent resolves to nothing
            if "symbology_invalid_request" not in str(exc):
                raise
            pd.DataFrame().to_parquet(target)
            written += 1
            empty += 1
            records.append(
                {
                    "session": session.isoformat(),
                    "cost_estimate_usd": 0.0,
                    "chain_rows": 0,
                    "zero_dte_rows": 0,
                    "zero_dte_contracts": 0,
                    "note": "no SPXW symbols resolved; non-trading day",
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
            continue
        if spent + cost > cap:
            print(f"HALT: {session} would take spend to {spent + cost:.4f} over cap {cap}")
            break
        spent += cost
        frame = _retry(
            f"{session} download",
            lambda: client.timeseries.get_range(
                dataset=DATASET, schema=SCHEMA, symbols=[PARENT],
                stype_in="parent", start=start, end=end,
            ),
        ).to_df()
        zero = _zero_dte(frame, session)
        zero.to_parquet(target)
        written += 1
        if zero.empty:
            empty += 1
        records.append(
            {
                "session": session.isoformat(),
                "cost_estimate_usd": cost,
                "chain_rows": int(len(frame)),
                "zero_dte_rows": int(len(zero)),
                "zero_dte_contracts": int(zero["symbol"].nunique()) if len(zero) else 0,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        if i % 25 == 0:
            print(
                f"  {i}/{len(sessions)} sessions, {written} written, "
                f"{empty} without a same-day expiry, ${spent:.4f}",
                flush=True,
            )

    receipt = args.root / "acquisition_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "v5.spxw-0dte-acquisition.v1",
                "acquired_on": "2026-08-13",
                "dataset": DATASET, "schema": SCHEMA, "parent": PARENT,
                "window": "09:30-16:00 America/New_York, DST-aware",
                "span": [args.start, args.end],
                "weekdays_in_span": len(sessions),
                "sessions_written": written,
                "sessions_without_same_day_expiry": empty,
                "estimated_spend_usd": round(spent, 6),
                "cap_usd": cap,
                "manifest": str(args.manifest),
                "zero_dte_rule": (
                    "a contract counts as 0DTE only when the expiry encoded in its "
                    "own OSI symbol equals the session date; membership is derived, "
                    "never assumed from a calendar"
                ),
                "sessions": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwritten {written}, empty {empty}, spend ${spent:.4f} of ${cap:.2f}")
    print(f"receipt: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
