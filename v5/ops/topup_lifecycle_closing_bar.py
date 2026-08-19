"""Acquire the one closing bar the backfill request window excluded.

**The defect this repairs.** `download_spxw_history._bounds` builds the `cbbo-1m`
window as ``09:30 -> 16:00`` ET and vendor time ranges are half-open
``[start, end)``. CBBO-1m bars are stamped at their **end** in ``ts_recv``, so the
bar stamped ``16:00:00`` -- the one covering event minute **15:59** -- sits exactly
on the exclusive bound and was never delivered. Measured on the acquired corpus:
389 bars per contract instead of 390, `ts_recv` 09:31->15:59, and 0 of 62 sampled
sessions carry a 16:00 row. The owned corpus, acquired under a wider window, does
carry it, and on 2025-08-01 that bar held **more** quoting contracts than the
15:59 bar (500 against 413), so this is the closing-auction minute rather than a
sleepy one. It is also the exact minute the dataset law reads for terminal
accounting (``terminal = rth[minute == LAST_QUOTE_MINUTE]``).

**Why a separate tool.** `acquire_lifecycle_backfill` is pinned by
`implementation_sha256` inside declaration V4 and its passing preflight. Editing
it to widen the window would invalidate both. This tool leaves that module,
its receipt and its hashes untouched and writes to its own destination.

**What it does not do.** It resolves nothing from symbology -- the authoritative
same-day ladder is already on disk in each session's `definition` parquet, so the
top-up costs no symbology calls. It re-applies the charter's same-day rule at
request time by filtering each ladder on its own OSI expiry, refuses to overwrite
any existing file, prices every session before requesting any data, and enforces
its ceiling twice: as a preflight gate and as a running total in the download
loop.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.acquire_lifecycle_backfill import (
    CBBO_SCHEMA,
    _payload_sha256,
    _with_retry,
)
from v5.ops.download_spxw_history import (
    DATASET,
    NY,
    OSI,
    _load_env,
    canonical_json,
    file_sha256,
)

SCHEMA_VERSION = "v5.lifecycle-closing-bar-topup.v1"
# The bar we are missing is stamped exactly at the close. Vendor filtering for
# this schema acts on the bar's own `ts_recv`, established from the delivered
# data: a [09:30, 16:00) request returned ts_recv 09:31..15:59, which is only
# consistent with the stamp being filtered, not the events inside the bar.
TARGET_STAMP = time(16, 0)


class TopupError(RuntimeError):
    """The closing-bar top-up cannot proceed safely."""


def _window(session: str, *, seconds: int) -> tuple[str, str]:
    """Half-open window that contains exactly the 16:00:00 bar stamp."""

    day = date.fromisoformat(session)
    start = datetime.combine(day, TARGET_STAMP, tzinfo=NY)
    end = start + timedelta(seconds=seconds)
    return (
        start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def session_ladder(definition_root: Path, session: str) -> list[str]:
    """The session's same-day ladder, read from its already-owned definitions.

    The same-day rule is re-applied here from each symbol's own OSI expiry, so a
    definition file carrying anything else cannot widen the request.
    """

    path = definition_root / f"{session}.definition.parquet"
    if not path.is_file():
        raise TopupError(f"definition file missing for {session}: {path}")
    frame = pd.read_parquet(path)
    if "raw_symbol" not in frame.columns:
        frame = frame.reset_index()
    if "raw_symbol" not in frame.columns:
        raise TopupError(f"definition file has no raw_symbol column: {path}")
    stamp = date.fromisoformat(session).strftime("%y%m%d")
    ladder = sorted(
        {
            str(symbol)
            for symbol in frame["raw_symbol"].astype(str).unique()
            if (match := OSI.match(str(symbol))) and match.group("expiry") == stamp
        }
    )
    if not ladder:
        raise TopupError(f"no same-day SPXW symbol in the definitions for {session}")
    return ladder


def _cost(client: Any, *, session: str, symbols: list[str], seconds: int) -> float:
    start, end = _window(session, seconds=seconds)
    value = float(
        _with_retry(
            f"topup cost {session}",
            lambda: client.metadata.get_cost(
                dataset=DATASET,
                schema=CBBO_SCHEMA,
                symbols=symbols,
                stype_in="raw_symbol",
                start=start,
                end=end,
            ),
        )
    )
    if value < 0.0:
        raise TopupError(f"vendor returned a negative cost for {session}")
    return value


def sessions_from_receipt(receipt_path: Path) -> list[str]:
    """The authoritative session list: what the acquisition actually recorded."""

    if not receipt_path.is_file():
        raise TopupError(f"acquisition receipt missing: {receipt_path}")
    payload = json.loads(receipt_path.read_text())
    if payload.get("receipt_sha256") != _payload_sha256(payload, "receipt_sha256"):
        raise TopupError("acquisition receipt self-hash mismatch")
    sessions = sorted(
        {
            str(item["session"])
            for item in payload.get("files", [])
            if item.get("schema") == CBBO_SCHEMA
        }
    )
    if not sessions:
        raise TopupError("acquisition receipt records no cbbo sessions")
    return sessions


def write_preflight(
    *,
    receipt_path: Path,
    definition_root: Path,
    output_path: Path,
    cap_usd: float,
    seconds: int,
    client: Any,
) -> dict[str, Any]:
    if output_path.exists():
        raise TopupError(f"refusing to overwrite preflight receipt: {output_path}")
    sessions = sessions_from_receipt(receipt_path)
    rows: list[dict[str, Any]] = []
    total = 0.0
    for index, session in enumerate(sessions, start=1):
        ladder = session_ladder(definition_root, session)
        usd = _cost(client, session=session, symbols=ladder, seconds=seconds)
        total += usd
        rows.append({"session": session, "symbols": len(ladder), "cbbo_usd": usd})
        if index % 50 == 0 or index == len(sessions):
            print(f"priced {index}/{len(sessions)} sessions; ${total:.6f}", flush=True)
    start, end = _window(sessions[0], seconds=seconds)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "phase": "preflight",
        "purpose": (
            "price the closing bar (ts_recv 16:00:00) that the 09:30-16:00 half-open "
            "backfill window excluded from all 794 acquired sessions"
        ),
        "repairs": {
            "defect": "half-open [09:30,16:00) window drops the end-stamped 16:00:00 bar",
            "measured": "389 bars/contract delivered; ts_recv 09:31-15:59; 0/62 sampled sessions carry 16:00",
            "source_receipt": str(receipt_path),
        },
        "request": {
            "dataset": DATASET,
            "schema": CBBO_SCHEMA,
            "stype_in": "raw_symbol",
            "window_seconds": seconds,
            "window_example": {"session": sessions[0], "start": start, "end": end},
            "ladder_source": "each session's owned definition parquet, filtered by its own OSI expiry",
            "same_day_rule": "a requested symbol's own OSI YYMMDD must equal its session",
        },
        "sessions": len(sessions),
        "hard_cap_usd": cap_usd,
        "estimated_total_usd": total,
        "gate": "PASS" if total <= cap_usd else "STOP_OVER_HARD_CAP",
        "session_costs": rows,
        "integrity": {
            "download_performed": False,
            "money_spent": False,
            "broker_contacted": False,
            "reserved_sessions_used": False,
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt, "receipt_sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _verify_topup(frame: pd.DataFrame, session: str) -> dict[str, Any]:
    """A saved top-up must contain only the closing bar, only same-day symbols."""

    flat = frame.reset_index()
    if "ts_recv" not in flat.columns:
        raise TopupError(f"{session}: top-up frame has no ts_recv")
    stamps = (
        pd.to_datetime(flat["ts_recv"], utc=True)
        .dt.tz_convert(NY)
        .dt.strftime("%H:%M:%S")
        .unique()
        .tolist()
    )
    if stamps != ["16:00:00"]:
        raise TopupError(f"{session}: expected only the 16:00:00 bar, got {sorted(stamps)}")
    expiry = date.fromisoformat(session).strftime("%y%m%d")
    symbols = flat["symbol"].astype(str).unique().tolist()
    bad = [s for s in symbols if not ((m := OSI.match(s)) and m.group("expiry") == expiry)]
    if bad:
        raise TopupError(f"{session}: non-same-day symbol in top-up: {sorted(bad)[:3]}")
    return {"session": session, "rows": int(len(flat)), "contracts": int(len(symbols))}


def acquire(
    *,
    receipt_path: Path,
    definition_root: Path,
    preflight_path: Path,
    out_root: Path,
    acquisition_out: Path,
    seconds: int,
    client: Any,
) -> dict[str, Any]:
    if acquisition_out.exists():
        raise TopupError(f"refusing to overwrite top-up receipt: {acquisition_out}")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("receipt_sha256") != _payload_sha256(preflight, "receipt_sha256"):
        raise TopupError("top-up preflight self-hash mismatch")
    if preflight.get("gate") != "PASS":
        raise TopupError(f"top-up preflight did not pass: {preflight.get('gate')}")
    if preflight.get("implementation_sha256") != file_sha256(Path(__file__)):
        raise TopupError("preflight pins a different tool than the code running")
    cap = float(preflight["hard_cap_usd"])
    quoted = {row["session"]: float(row["cbbo_usd"]) for row in preflight["session_costs"]}

    out_root.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    saved: list[dict[str, Any]] = []
    for index, session in enumerate(sorted(quoted), start=1):
        path = out_root / f"{session}.cbbo-1m.closing.parquet"
        if path.exists():  # idempotent resume; never re-buy
            saved.append({"session": session, "path": str(path), "skipped": True})
            continue
        if spent + quoted[session] > cap:
            raise TopupError(f"running spend would exceed the ceiling at {session}")
        ladder = session_ladder(definition_root, session)
        start, end = _window(session, seconds=seconds)
        data = _with_retry(
            f"topup download {session}",
            lambda: client.timeseries.get_range(
                dataset=DATASET,
                schema=CBBO_SCHEMA,
                symbols=ladder,
                stype_in="raw_symbol",
                start=start,
                end=end,
            ),
        )
        frame = data.to_df()
        if frame.empty:
            raise TopupError(f"{session}: vendor returned no closing bar for the declared window")
        checked = _verify_topup(frame, session)
        frame.to_parquet(path)
        spent += quoted[session]
        saved.append({**checked, "path": str(path), "skipped": False})
        if index % 50 == 0 or index == len(quoted):
            print(f"topped up {index}/{len(quoted)}; ${spent:.6f}", flush=True)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "phase": "acquire",
        "preflight_sha256": preflight["receipt_sha256"],
        "hard_cap_usd": cap,
        "quoted_total_usd": sum(quoted.values()),
        "spent_this_run_usd": spent,
        "destination": str(out_root),
        "completion": {
            "sessions": len(quoted),
            "downloaded": sum(0 if item.get("skipped") else 1 for item in saved),
            "skipped_existing": sum(1 if item.get("skipped") else 0 for item in saved),
            "contracts_total": sum(int(item.get("contracts", 0)) for item in saved),
            "rows_total": sum(int(item.get("rows", 0)) for item in saved),
        },
        "verification": (
            "every saved file asserted to contain only the 16:00:00 ts_recv stamp "
            "and only symbols whose own OSI expiry equals the session"
        ),
        "files": saved,
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt, "receipt_sha256")
    acquisition_out.parent.mkdir(parents=True, exist_ok=True)
    acquisition_out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "acquire"), required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--definition-root", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--topup-out", type=Path, default=None)
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--window-seconds", type=int, default=1)
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    args = parser.parse_args()

    _load_env(args.env_file)
    import databento as db

    client = db.Historical(os.environ.get("DATABENTO_API_KEY"))
    if args.phase == "preflight":
        receipt = write_preflight(
            receipt_path=args.acquisition_receipt,
            definition_root=args.definition_root,
            output_path=args.preflight_out,
            cap_usd=args.cap_usd,
            seconds=args.window_seconds,
            client=client,
        )
        print(
            f"closing-bar top-up preflight {receipt['gate']}: "
            f"${receipt['estimated_total_usd']:.6f} of ${receipt['hard_cap_usd']:.2f} "
            f"over {receipt['sessions']} sessions"
        )
        return 0 if receipt["gate"] == "PASS" else 3
    if args.out_root is None or args.topup_out is None:
        raise SystemExit("--out-root and --topup-out are required for --phase acquire")
    receipt = acquire(
        receipt_path=args.acquisition_receipt,
        definition_root=args.definition_root,
        preflight_path=args.preflight_out,
        out_root=args.out_root,
        acquisition_out=args.topup_out,
        seconds=args.window_seconds,
        client=client,
    )
    print(
        f"closing bar acquired for {receipt['completion']['downloaded']} sessions; "
        f"${receipt['spent_this_run_usd']:.6f} spent"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
