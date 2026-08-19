"""Guarded acquisition of the declared SPXW 0DTE historical quote backfill.

The runner has two deliberately separated phases:

``--phase preflight``
    Reads the frozen, pre-existing OHLCV inventory and asks Databento for the
    exact cost of every declaration member and schema.  It never downloads.

``--phase acquire``
    Requires that passing cost receipt and downloads only the declared missing
    files.  Every request is for ``SPXW.OPT`` and every saved row is filtered
    by its own OSI expiry, never by a calendar assumption.

The source OHLCV files supply an outcome-blind list of the days on which a
same-day SPXW contract actually traded.  They are not copied or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq


DATASET = "OPRA.PILLAR"
PARENT = "SPXW.OPT"
SCHEMAS = ("definition", "cbbo-1m")
NY = ZoneInfo("America/New_York")
OSI = re.compile(r"^SPXW\s+(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


class AcquisitionError(RuntimeError):
    """A declaration, preflight, or acquisition invariant failed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any], field: str) -> str:
    semantic = dict(payload)
    semantic.pop(field, None)
    return hashlib.sha256(canonical_json(semantic)).hexdigest()


def _verify_semantic_freeze(reference: dict[str, Any]) -> None:
    """Require the A8 source-and-semantics freeze before vendor contact."""

    path = Path(str(reference.get("path", "")))
    if not path.is_file():
        raise AcquisitionError("pre-acquisition semantic freeze is missing")
    freeze = json.loads(path.read_text())
    expected = str(reference.get("freeze_sha256", ""))
    actual = _payload_sha256(freeze, "freeze_sha256")
    if freeze.get("freeze_sha256") != actual or expected != actual:
        raise AcquisitionError("pre-acquisition semantic freeze self-hash mismatch")
    root = Path(__file__).resolve().parents[2]
    for relative, declared_hash in dict(freeze.get("source_files", {})).items():
        source = root / relative
        if not source.is_file() or file_sha256(source) != declared_hash:
            raise AcquisitionError(f"semantic freeze source drift: {relative}")


def _load_env(path: Path) -> None:
    """Load an optional local credentials file without printing credentials."""

    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _date_range(start: str, end: str) -> tuple[date, date]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise AcquisitionError("declaration start is after its end")
    return first, last


def source_sessions(
    root: Path,
    *,
    start: str,
    end: str,
) -> tuple[list[str], str]:
    """Return exactly the nonempty owned OHLCV sessions and their stable hash."""

    _date_range(start, end)
    sessions: list[str] = []
    for path in sorted(root.glob("*.spxw_0dte.ohlcv-1m.parquet")):
        session = path.name[:10]
        if not (start <= session <= end):
            continue
        try:
            count = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception as exc:  # pragma: no cover - filesystem-specific detail
            raise AcquisitionError(f"cannot read source OHLCV metadata: {path}: {exc}") from exc
        if count > 0:
            sessions.append(session)
    if not sessions:
        raise AcquisitionError("the declared source inventory has no nonempty sessions")
    return sessions, hashlib.sha256(canonical_json(sessions)).hexdigest()


def _bounds(session: str, schema: str) -> tuple[str, str]:
    """Use full UTC days for definitions and RTH ET for minute CBBO."""

    day = date.fromisoformat(session)
    if schema == "definition":
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
    elif schema == "cbbo-1m":
        base = datetime.combine(day, time.min, tzinfo=NY)
        start = base.replace(hour=9, minute=30).astimezone(timezone.utc)
        end = base.replace(hour=16, minute=0).astimezone(timezone.utc)
    else:
        raise AcquisitionError(f"unsupported declared schema: {schema}")
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def _zero_dte(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    """Keep only rows whose own OPRA symbol encodes that session's expiry."""

    if frame.empty or "symbol" not in frame.columns:
        return frame.iloc[0:0].copy()
    flat = frame.reset_index()
    parsed = flat["symbol"].astype(str).str.extract(OSI)
    keep = parsed["expiry"].eq(date.fromisoformat(session).strftime("%y%m%d")).fillna(False)
    out = flat.loc[keep].copy()
    if out.empty:
        return out
    out["strike"] = parsed.loc[keep, "strike"].astype(float).to_numpy() / 1000.0
    out["right"] = parsed.loc[keep, "right"].to_numpy()
    return out.reset_index(drop=True)


def _schema_output(root: Path, session: str, schema: str) -> Path:
    directory = root / "raw" / "databento" / f"opra_spxw_{schema.replace('-', '_')}"
    return directory / f"{session}.{schema}.parquet"


def _load_declaration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AcquisitionError(f"declaration is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "v5.lifecycle-quote-backfill-declaration.v1":
        raise AcquisitionError("unexpected acquisition declaration schema")
    if payload.get("declaration_sha256") != _payload_sha256(payload, "declaration_sha256"):
        raise AcquisitionError("acquisition declaration self-hash mismatch")
    _verify_semantic_freeze(dict(payload.get("semantic_freeze", {})))
    expected_code = payload.get("implementation_sha256")
    actual_code = file_sha256(Path(__file__))
    if expected_code != actual_code:
        raise AcquisitionError("acquisition declaration implementation hash mismatch")
    request = payload.get("request", {})
    if request.get("dataset") != DATASET or request.get("parent") != PARENT:
        raise AcquisitionError("dataset or parent drifted from the declaration")
    if tuple(request.get("schemas", [])) != SCHEMAS:
        raise AcquisitionError("schema family drifted from the declaration")
    source = payload.get("source_ohlcv_inventory", {})
    sessions, source_hash = source_sessions(
        Path(source.get("root", "")),
        start=str(source.get("start", "")),
        end=str(source.get("end", "")),
    )
    if int(source.get("nonempty_sessions", -1)) != len(sessions):
        raise AcquisitionError("source inventory session count drifted")
    if source.get("sessions_sha256") != source_hash:
        raise AcquisitionError("source inventory hash drifted")
    cap = float(payload.get("hard_cap_usd", -1.0))
    if not (0.0 < cap <= 75.0):
        raise AcquisitionError("declared hard cap is outside the owner-authorized range")
    return payload


def _cost_rows(client: Any, sessions: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        for schema in SCHEMAS:
            start, end = _bounds(session, schema)
            cost = float(
                client.metadata.get_cost(
                    dataset=DATASET,
                    schema=schema,
                    symbols=[PARENT],
                    stype_in="parent",
                    start=start,
                    end=end,
                )
            )
            if cost < 0.0:
                raise AcquisitionError(f"vendor returned a negative cost for {session} {schema}")
            rows.append(
                {
                    "session": session,
                    "schema": schema,
                    "start": start,
                    "end": end,
                    "cost_estimate_usd": cost,
                }
            )
    return rows


def write_preflight(
    *,
    declaration_path: Path,
    output_path: Path,
    client: Any,
) -> dict[str, Any]:
    """Write a complete exact-cost receipt before any data request."""

    if output_path.exists():
        raise AcquisitionError(f"refusing to overwrite preflight receipt: {output_path}")
    declaration = _load_declaration(declaration_path)
    source = declaration["source_ohlcv_inventory"]
    sessions, source_hash = source_sessions(
        Path(source["root"]), start=source["start"], end=source["end"]
    )
    rows = _cost_rows(client, sessions)
    total = float(sum(row["cost_estimate_usd"] for row in rows))
    cap = float(declaration["hard_cap_usd"])
    receipt: dict[str, Any] = {
        "schema_version": "v5.lifecycle-quote-backfill-cost-preflight.v1",
        "declaration_path": str(declaration_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "request": declaration["request"],
        "source_inventory": {
            "nonempty_sessions": len(sessions),
            "sessions_sha256": source_hash,
        },
        "session_schema_costs": rows,
        "estimated_total_usd": total,
        "hard_cap_usd": cap,
        "gate": "PASS" if total <= cap else "STOP_OVER_HARD_CAP",
        "integrity": {
            "download_performed": False,
            "money_spent": False,
            "reserved_sessions_used": False,
            "broker_contacted": False,
        },
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt, "receipt_sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
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
        raise AcquisitionError("passing preflight exceeds the declared cap")
    return receipt


def _download_one(client: Any, *, session: str, schema: str, root: Path) -> dict[str, Any]:
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
    start, end = _bounds(session, schema)
    store = client.timeseries.get_range(
        dataset=DATASET,
        schema=schema,
        symbols=[PARENT],
        stype_in="parent",
        start=start,
        end=end,
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
    *,
    declaration_path: Path,
    preflight_path: Path,
    receipt_path: Path,
    client: Any,
) -> dict[str, Any]:
    """Acquire every missing declared file and write an immutable receipt."""

    if receipt_path.exists():
        raise AcquisitionError(f"refusing to overwrite acquisition receipt: {receipt_path}")
    declaration = _load_declaration(declaration_path)
    _load_passing_preflight(preflight_path, declaration)
    source = declaration["source_ohlcv_inventory"]
    sessions, source_hash = source_sessions(
        Path(source["root"]), start=source["start"], end=source["end"]
    )
    root = Path(declaration["destination"]["root"])
    rows = []
    for index, session in enumerate(sessions, 1):
        for schema in SCHEMAS:
            rows.append(_download_one(client, session=session, schema=schema, root=root))
        if index % 25 == 0 or index == len(sessions):
            print(f"downloaded/verified {index}/{len(sessions)} sessions", flush=True)
    receipt: dict[str, Any] = {
        "schema_version": "v5.lifecycle-quote-backfill-acquisition.v1",
        "declaration_path": str(declaration_path),
        "declaration_sha256": declaration["declaration_sha256"],
        "preflight_path": str(preflight_path),
        "preflight_sha256": file_sha256(preflight_path),
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_inventory": {
            "nonempty_sessions": len(sessions),
            "sessions_sha256": source_hash,
        },
        "destination": declaration["destination"],
        "files": rows,
        "completion": {
            "expected_files": len(sessions) * len(SCHEMAS),
            "recorded_files": len(rows),
            "all_rows_filter_by_own_osi_expiry": True,
        },
        "hard_stops": {
            "broker_accessed": False,
            "orders_submitted": False,
            "reserved_sessions_used": False,
            "model_loaded_or_fit": False,
            "spend_above_hard_cap": False,
        },
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt, "receipt_sha256")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
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
            declaration_path=args.declaration,
            output_path=args.preflight_out,
            client=client,
        )
        print(
            f"preflight {receipt['gate']}: ${receipt['estimated_total_usd']:.6f} "
            f"of ${receipt['hard_cap_usd']:.2f}; {len(receipt['session_schema_costs'])} requests"
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
