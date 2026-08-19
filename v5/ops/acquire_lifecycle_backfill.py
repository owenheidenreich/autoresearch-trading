"""Corrected same-day SPXW quote backfill: resolve the 0DTE ladder, then price it.

The 2026-08-15 preflight priced `stype_in="parent"` (`SPXW.OPT`) for `cbbo-1m`,
which covers **every SPXW expiration listed that day** — 12,828 to 15,980
instruments measured — and returned $671.90 against the $75 ceiling. It STOPPED
correctly. But the development charter authorizes only "SPXW contracts whose own
OSI expiry equals each session date", so that request was **broader than the
owner approved** and its price was not the price of the authorized data.

This runner requests exactly the authorized scope:

1. resolve the whole SPXW universe for the session (symbology, no data, free);
2. keep only symbols whose own OSI expiry equals the session — the same-day rule
   applied at **request** time rather than after paying for the discard;
3. price `definition` (parent scope, measured $0.00) and `cbbo-1m` (the resolved
   0DTE ladder) with the vendor's exact cost endpoint; and
4. download only if the exact total is at or below the declared ceiling.

Measured on the full ladder: 362-988 contracts and $0.0215-$0.0483 per session,
against $0.85 at parent scope.

The ceiling is enforced twice — once as the preflight gate, and again as a
running total inside the acquisition loop, so no sequence of per-session
requests can walk past it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from v5.ops.download_spxw_history import (
    DATASET,
    NY,
    OSI,
    PARENT,
    AcquisitionError,
    _bounds,
    _load_env,
    _schema_output,
    _zero_dte,
    canonical_json,
    file_sha256,
    source_sessions,
)

DECLARATION_SCHEMA = "v5.lifecycle-quote-backfill-declaration.v2"
DEFINITION_SCHEMA = "definition"
CBBO_SCHEMA = "cbbo-1m"


VENDOR_ATTEMPTS = 5
VENDOR_BACKOFF_SECONDS = (5.0, 15.0, 45.0, 120.0)


def _payload_sha256(payload: dict[str, Any], field: str) -> str:
    unsigned = {k: v for k, v in payload.items() if k != field}
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _with_retry(what: str, call):
    """Run one vendor call, retrying transient server-side failures.

    Measured 2026-08-16: the vendor intermittently returns `504 gateway
    timed out` and ends streams early, and a single such failure previously
    killed a whole acquisition attempt. Retrying inside the call means one blip
    no longer discards the work already done in that attempt.

    Only transport-level failures are retried. A refusal by this module's own
    guards — the ceiling, ladder drift, a declaration mismatch — is a decision
    and is raised immediately.
    """

    last: Exception | None = None
    for attempt in range(VENDOR_ATTEMPTS):
        try:
            return call()
        except AcquisitionError:
            raise
        except Exception as exc:  # vendor transport errors are not a stable class
            last = exc
            if attempt + 1 == VENDOR_ATTEMPTS:
                break
            delay = VENDOR_BACKOFF_SECONDS[min(attempt, len(VENDOR_BACKOFF_SECONDS) - 1)]
            print(f"  {what}: {type(exc).__name__}; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise AcquisitionError(f"{what} failed after {VENDOR_ATTEMPTS} attempts: {last}")


def load_declaration(path: Path) -> dict[str, Any]:
    """Verify the declaration's self-hash, pinned runner, and ceiling."""

    if not path.is_file():
        raise AcquisitionError(f"declaration is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != DECLARATION_SCHEMA:
        raise AcquisitionError("declaration schema mismatch")
    if payload.get("declaration_sha256") != _payload_sha256(payload, "declaration_sha256"):
        raise AcquisitionError("declaration self-hash mismatch")
    if payload.get("implementation_sha256") != file_sha256(Path(__file__)):
        raise AcquisitionError("declaration pins a different runner than the code running")
    cap = float(payload.get("hard_cap_usd", -1.0))
    if not (0.0 < cap <= 75.0):
        raise AcquisitionError("declared ceiling is outside the owner-authorized range")
    request = payload.get("request", {})
    if request.get("cbbo_stype_in") != "raw_symbol":
        raise AcquisitionError("declaration must request cbbo at resolved-symbol scope")
    close = request.get("cbbo_close_minute")
    if not isinstance(close, str) or not re.fullmatch(r"\d{2}:\d{2}", close):
        raise AcquisitionError(
            "declaration must state cbbo_close_minute as HH:MM; the half-open window "
            "drops the bar stamped at it, so the close cannot be an implicit constant"
        )
    # The pinned runner hash covers this file only, but the request window used to
    # come from `download_spxw_history._bounds` — so the window could have changed
    # without invalidating the pin. Any declaration that names a window dependency
    # must have its bytes verified too.
    for dependency in payload.get("implementation_dependencies", []):
        pinned = Path(dependency["path"])
        if not pinned.is_file() or file_sha256(pinned) != dependency["sha256"]:
            raise AcquisitionError(
                f"declared implementation dependency drifted: {dependency['path']}"
            )
    return payload


def resolve_zero_dte_ladder(client: Any, session: str) -> list[str]:
    """Every SPXW symbol whose own OSI expiry equals this session.

    Symbology only: no market data is requested and nothing is charged. The
    membership test reads each symbol's own expiry, never a trading calendar.
    """

    day = date.fromisoformat(session)
    resolved = _with_retry(
        f"symbology {session}",
        lambda: client.symbology.resolve(
            dataset=DATASET,
            symbols=[PARENT],
            stype_in="parent",
            stype_out="instrument_id",
            start_date=session,
            end_date=(day + timedelta(days=1)).isoformat(),
        ),
    )
    universe = resolved.get("result", {})
    stamp = day.strftime("%y%m%d")
    ladder = sorted(
        symbol
        for symbol in universe
        if (match := OSI.match(str(symbol))) and match.group("expiry") == stamp
    )
    if not ladder:
        raise AcquisitionError(f"no same-day SPXW ladder resolved for {session}")
    return ladder


def _window(session: str, schema: str, *, cbbo_close: str) -> tuple[str, str]:
    """Declared request window; the CBBO close is data, not a constant.

    Vendor ranges are half-open `[start, end)` and CBBO-1m bars are stamped at
    their **end** in `ts_recv`, so a window ending at the declared close drops
    the bar stamped exactly at it. Measured on the V4 acquisition: a
    09:30->16:00 request delivered `ts_recv` 09:31-15:59 — 389 bars per contract
    instead of 390 — and 0 of 62 sampled sessions carried a 16:00 row, which is
    the minute the dataset law reads for terminal accounting. The close is
    therefore declared per run and must be set past the last bar wanted.

    `definition` keeps its full-UTC-day window; only `cbbo-1m` is affected.
    """

    if schema == DEFINITION_SCHEMA:
        return _bounds(session, schema)
    if schema != CBBO_SCHEMA:
        raise AcquisitionError(f"unsupported declared schema: {schema}")
    hour, minute = (int(part) for part in cbbo_close.split(":"))
    day = date.fromisoformat(session)
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    end = datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(timezone.utc)
    if end <= start:
        raise AcquisitionError(f"declared cbbo close {cbbo_close} is not after the open")
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def _cost(
    client: Any,
    *,
    schema: str,
    symbols: list[str],
    stype_in: str,
    session: str,
    cbbo_close: str,
) -> float:
    start, end = _window(session, schema, cbbo_close=cbbo_close)
    value = float(
        _with_retry(
            f"cost {session} {schema}",
            lambda: client.metadata.get_cost(
                dataset=DATASET,
                schema=schema,
                symbols=symbols,
                stype_in=stype_in,
                start=start,
                end=end,
            ),
        )
    )
    if value < 0.0:
        raise AcquisitionError(f"vendor returned a negative cost for {session} {schema}")
    return value


def session_costs(client: Any, session: str, *, cbbo_close: str) -> dict[str, Any]:
    """Exact vendor cost for one session at the authorized scope."""

    ladder = resolve_zero_dte_ladder(client, session)
    definition_usd = _cost(
        client,
        schema=DEFINITION_SCHEMA,
        symbols=[PARENT],
        stype_in="parent",
        session=session,
        cbbo_close=cbbo_close,
    )
    cbbo_usd = _cost(
        client,
        schema=CBBO_SCHEMA,
        symbols=ladder,
        stype_in="raw_symbol",
        session=session,
        cbbo_close=cbbo_close,
    )
    return {
        "session": session,
        "ladder_symbols": len(ladder),
        "definition_usd": definition_usd,
        "cbbo_usd": cbbo_usd,
        "session_usd": definition_usd + cbbo_usd,
    }


def write_preflight(*, declaration_path: Path, output_path: Path, client: Any) -> dict[str, Any]:
    """Exact cost for every declared session before any data is requested."""

    if output_path.exists():
        raise AcquisitionError(f"refusing to overwrite preflight receipt: {output_path}")
    declaration = load_declaration(declaration_path)
    cbbo_close = declaration["request"]["cbbo_close_minute"]
    inventory = declaration["source_ohlcv_inventory"]
    sessions, sessions_hash = source_sessions(
        Path(inventory["root"]), start=inventory["start"], end=inventory["end"]
    )
    if sessions_hash != inventory["sessions_sha256"]:
        raise AcquisitionError("source inventory hash drifted from the declaration")

    # Checkpoint each priced session. The preflight makes roughly 2,400 vendor
    # calls; without this a single failure late in the pass discards the whole
    # hour. Costing is free, so a resumed pass buys nothing extra.
    checkpoint = output_path.with_suffix(".partial.jsonl")
    priced: dict[str, dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                priced[str(row["session"])] = row
        print(f"resuming preflight from {len(priced)} checkpointed sessions", flush=True)

    with checkpoint.open("a") as handle:
        for index, session in enumerate(sessions, start=1):
            if session in priced:
                continue
            row = session_costs(client, session, cbbo_close=cbbo_close)
            priced[session] = row
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if index % 50 == 0:
                running = sum(r["session_usd"] for r in priced.values())
                print(f"  priced {index}/{len(sessions)} — ${running:.4f}", flush=True)

    rows = [priced[session] for session in sessions]
    total = sum(row["session_usd"] for row in rows)
    cap = float(declaration["hard_cap_usd"])
    receipt: dict[str, Any] = {
        "schema_version": "v5.lifecycle-quote-backfill-cost-preflight.v2",
        "declaration_path": str(declaration_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "supersedes": {
            "receipt": (
                "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/cost_preflight.json"
            ),
            "reason": (
                "that receipt priced parent scope — every listed SPXW expiration — which is "
                "broader than the charter-authorized same-day scope; its STOP stands as the "
                "correct decision on the request it actually priced"
            ),
        },
        "request": declaration["request"],
        "source_inventory": {"sessions": len(sessions), "sessions_sha256": sessions_hash},
        "hard_cap_usd": cap,
        "estimated_total_usd": total,
        "definition_total_usd": sum(row["definition_usd"] for row in rows),
        "cbbo_total_usd": sum(row["cbbo_usd"] for row in rows),
        "gate": "PASS" if total <= cap else "STOP_OVER_HARD_CAP",
        "integrity": {
            "money_spent": False,
            "download_performed": False,
            "broker_contacted": False,
            "reserved_sessions_used": False,
        },
        "session_costs": rows,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _load_passing_preflight(path: Path, declaration: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise AcquisitionError(f"preflight receipt is missing: {path}")
    receipt = json.loads(path.read_text())
    if receipt.get("receipt_sha256") != _payload_sha256(receipt, "receipt_sha256"):
        raise AcquisitionError("preflight receipt self-hash mismatch")
    if receipt.get("gate") != "PASS":
        raise AcquisitionError("preflight did not pass; acquisition is refused")
    if receipt.get("declaration_sha256") != declaration.get("declaration_sha256"):
        raise AcquisitionError("preflight declaration differs from acquisition declaration")
    if receipt.get("implementation_sha256") != file_sha256(Path(__file__)):
        raise AcquisitionError("preflight implementation differs from acquisition code")
    if float(receipt.get("estimated_total_usd", float("inf"))) > float(declaration["hard_cap_usd"]):
        raise AcquisitionError("passing preflight exceeds the declared ceiling")
    return receipt


def _download_one(
    client: Any,
    *,
    session: str,
    schema: str,
    root: Path,
    symbols: list[str],
    stype_in: str,
    cbbo_close: str,
) -> dict[str, Any]:
    path = _schema_output(root, session, schema)
    if path.exists():
        return {
            "session": session,
            "schema": schema,
            "status": "EXISTING_DECLARED_OUTPUT",
            "path": str(path),
            "sha256": file_sha256(path),
            "saved_rows": int(pq.ParquetFile(path).metadata.num_rows),
        }
    start, end = _window(session, schema, cbbo_close=cbbo_close)
    store = _with_retry(
        f"download {session} {schema}",
        lambda: client.timeseries.get_range(
            dataset=DATASET,
            schema=schema,
            symbols=symbols,
            stype_in=stype_in,
            start=start,
            end=end,
        ),
    )
    frame = store.to_df()
    filtered = _zero_dte(frame, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(path, index=True)
    return {
        "session": session,
        "schema": schema,
        "status": "DOWNLOADED",
        "path": str(path),
        "sha256": file_sha256(path),
        "vendor_rows": int(len(frame)),
        "saved_rows": int(len(filtered)),
        "saved_contracts": int(filtered["symbol"].nunique()) if len(filtered) else 0,
    }


def acquire(
    *, declaration_path: Path, preflight_path: Path, receipt_path: Path, client: Any
) -> dict[str, Any]:
    """Download the authorized scope, enforcing the ceiling per session."""

    if receipt_path.exists():
        raise AcquisitionError(f"refusing to overwrite acquisition receipt: {receipt_path}")
    declaration = load_declaration(declaration_path)
    cbbo_close = declaration["request"]["cbbo_close_minute"]
    preflight = _load_passing_preflight(preflight_path, declaration)
    cap = float(declaration["hard_cap_usd"])
    root = Path(declaration["destination"]["root"])
    quoted = {row["session"]: row for row in preflight["session_costs"]}

    files: list[dict[str, Any]] = []
    spent = 0.0
    for session, row in quoted.items():
        # Defense in depth: the gate already passed on the total, but a running
        # check means no per-session sequence can walk past the ceiling. It
        # counts every declared session, complete or not, so the ceiling still
        # governs the job as a whole across restarts.
        spent += float(row["session_usd"])
        if spent > cap:
            raise AcquisitionError(
                f"running spend {spent:.4f} would exceed the ceiling at {session}"
            )
        # A session whose outputs both exist is already bought. Resolving its
        # ladder again buys nothing and costs a vendor round trip: before this
        # check a restart made ~800 symbology calls before reaching new work,
        # so one transient 504 anywhere in that pass discarded the whole
        # attempt. Measured 2026-08-16 at 3-29 s per resolve.
        existing = [
            _schema_output(root, session, schema)
            for schema in (DEFINITION_SCHEMA, CBBO_SCHEMA)
        ]
        if all(path.exists() for path in existing):
            for schema, path in zip((DEFINITION_SCHEMA, CBBO_SCHEMA), existing):
                files.append(
                    {
                        "session": session,
                        "schema": schema,
                        "status": "EXISTING_DECLARED_OUTPUT",
                        "path": str(path),
                        "sha256": file_sha256(path),
                        "saved_rows": int(pq.ParquetFile(path).metadata.num_rows),
                    }
                )
            continue
        ladder = resolve_zero_dte_ladder(client, session)
        if len(ladder) != int(row["ladder_symbols"]):
            raise AcquisitionError(
                f"ladder for {session} changed since the preflight "
                f"({len(ladder)} vs {row['ladder_symbols']}); acquisition refused"
            )
        files.append(
            _download_one(
                client,
                session=session,
                schema=DEFINITION_SCHEMA,
                root=root,
                symbols=[PARENT],
                stype_in="parent",
                cbbo_close=cbbo_close,
            )
        )
        files.append(
            _download_one(
                client,
                session=session,
                schema=CBBO_SCHEMA,
                root=root,
                symbols=ladder,
                stype_in="raw_symbol",
                cbbo_close=cbbo_close,
            )
        )

    receipt: dict[str, Any] = {
        "schema_version": "v5.lifecycle-quote-backfill-acquisition.v2",
        "declaration_sha256": declaration["declaration_sha256"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "preflight_sha256": preflight["receipt_sha256"],
        "quoted_total_usd": float(preflight["estimated_total_usd"]),
        "hard_cap_usd": cap,
        "completion": {
            "sessions": len(quoted),
            "recorded_files": len(files),
            "downloaded": sum(1 for f in files if f["status"] == "DOWNLOADED"),
        },
        "files": files,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "acquire"), required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--acquisition-out", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    args = parser.parse_args()

    _load_env(args.env_file)
    import databento as db

    client = db.Historical(os.environ.get("DATABENTO_API_KEY"))
    if args.phase == "preflight":
        receipt = write_preflight(
            declaration_path=args.declaration, output_path=args.preflight_out, client=client
        )
        print(
            f"preflight {receipt['gate']}: ${receipt['estimated_total_usd']:.4f}"
            f" of ${receipt['hard_cap_usd']:.2f}"
            f" (definitions ${receipt['definition_total_usd']:.4f},"
            f" cbbo ${receipt['cbbo_total_usd']:.4f})"
            f" over {receipt['source_inventory']['sessions']} sessions"
        )
        return 0 if receipt["gate"] == "PASS" else 3
    if args.acquisition_out is None:
        raise SystemExit("--acquisition-out is required for --phase acquire")
    receipt = acquire(
        declaration_path=args.declaration,
        preflight_path=args.preflight_out,
        receipt_path=args.acquisition_out,
        client=client,
    )
    print(f"acquired {receipt['completion']['recorded_files']} declared files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
