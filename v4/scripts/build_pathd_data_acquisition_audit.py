"""Build the evidence bundle for the Path-D 12-month data corpus.

The corpus itself may live outside the repository. This command reads file and
Parquet metadata, hashes every corpus file, and writes only compact audit
artifacts into the repository quarantine directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
OPTION_DIRS = {
    "definition": "raw/databento/opra_spxw_definition",
    "cbbo-1m": "raw/databento/opra_spxw_cbbo_1m",
    "ohlcv-1m": "raw/databento/opra_spxw_ohlcv_1m",
    "statistics": "raw/databento/opra_spxw_statistics",
    "cbbo-1s": "raw/databento/opra_spxw_cbbo_1s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def _date(path: Path) -> str | None:
    match = DATE_RE.search(path.name)
    return match.group(1) if match else None


def _parquet_summary(path: Path, pattern: str = "*.parquet") -> dict[str, Any]:
    files = sorted(path.glob(pattern)) if path.exists() else []
    dates = sorted({value for file in files if (value := _date(file))})
    rows_by_file = {
        file: pq.ParquetFile(file).metadata.num_rows for file in files
    }
    nonempty_dates = sorted(
        {
            value
            for file, rows in rows_by_file.items()
            if rows > 0 and (value := _date(file))
        }
    )
    return {
        "path": str(path),
        "files": len(files),
        "rows": sum(rows_by_file.values()),
        "session_dates": dates,
        "session_count": len(dates),
        "nonempty_session_dates": nonempty_dates,
        "nonempty_session_count": len(nonempty_dates),
        "empty_files": sum(rows == 0 for rows in rows_by_file.values()),
        "first_session": dates[0] if dates else None,
        "last_session": dates[-1] if dates else None,
    }


def _pickle_summary(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.pkl")) if path.exists() else []
    dates = sorted({value for file in files if (value := _date(file))})
    return {
        "path": str(path),
        "files": len(files),
        "rows": sum(len(pd.read_pickle(file)) for file in files),
        "session_dates": dates,
        "session_count": len(dates),
        "first_session": dates[0] if dates else None,
        "last_session": dates[-1] if dates else None,
    }


def _normalized_summary(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.parquet")) if path.exists() else []
    official = [file for file in files if "_official_context" in file.stem]
    base = [file for file in files if "_official_context" not in file.stem]

    def summarize(subset: list[Path]) -> dict[str, Any]:
        dates = sorted({value for file in subset if (value := _date(file))})
        return {
            "files": len(subset),
            "rows": sum(pq.ParquetFile(file).metadata.num_rows for file in subset),
            "session_dates": dates,
            "session_count": len(dates),
            "first_session": dates[0] if dates else None,
            "last_session": dates[-1] if dates else None,
        }

    return {"path": str(path), "base": summarize(base), "official_context": summarize(official)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_integrity_manifest(root: Path, out: Path) -> dict[str, Any]:
    files = sorted(file for file in root.rglob("*") if file.is_file())
    total_bytes = 0
    with out.open("w") as handle:
        for file in files:
            size = file.stat().st_size
            total_bytes += size
            handle.write(
                json.dumps(
                    {
                        "relative_path": str(file.relative_to(root)),
                        "bytes": size,
                        "sha256": _sha256(file),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {"files": len(files), "bytes": total_bytes, "manifest": str(out)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    root = args.corpus_root.resolve()
    audit = args.audit_dir.resolve()
    audit.mkdir(parents=True, exist_ok=True)

    option = {name: _parquet_summary(root / rel) for name, rel in OPTION_DIRS.items()}
    option_sets = {name: set(item["session_dates"]) for name, item in option.items()}
    canonical = option_sets["cbbo-1s"]
    spx = _parquet_summary(root / "vendor/thetadata/index/spx_1m")
    vix = _parquet_summary(root / "vendor/thetadata/index/vix_1m")
    minute = _pickle_summary(root / "aligned/processed/minute_entry")
    normalized = _normalized_summary(root / "aligned/normalized")
    es = _parquet_summary(root / "raw/databento/glbx_es_ohlcv_1m")
    vx = _parquet_summary(root / "raw/databento/xcbf_vx_ohlcv_1m")
    es_proxy = _parquet_summary(root / "raw/index/spx_1m", "*.proxy_es_fut.parquet")
    vx_proxy = _parquet_summary(root / "raw/index/vix_1m", "*.proxy_vx_fut.parquet")

    all_files = [file for file in root.rglob("*") if file.is_file()]
    dbn_files = [str(file) for file in all_files if ".dbn" in file.name]
    forbidden = [
        str(file)
        for file in all_files
        if "cmbp-1" in str(file).lower() or "tcbbo" in str(file).lower()
    ]
    checks = {
        "option_schema_session_sets_equal": all(values == canonical for values in option_sets.values()),
        "option_session_count_is_251": len(canonical) == 251,
        "official_spx_pairs_all_option_sessions": canonical <= set(spx["session_dates"]),
        "official_vix_pairs_all_option_sessions": canonical <= set(vix["session_dates"]),
        "minute_entry_pairs_all_option_sessions": canonical == set(minute["session_dates"]),
        "normalized_official_pairs_all_option_sessions": canonical
        == set(normalized["official_context"]["session_dates"]),
        "es_futures_context_present": es["session_count"] > 0,
        "vx_futures_context_present_from_vendor_boundary": vx["session_count"] > 0,
        "es_proxy_sessions_match_nonempty_raw": set(es["nonempty_session_dates"])
        == set(es_proxy["session_dates"]),
        "vx_proxy_sessions_match_nonempty_raw": set(vx["nonempty_session_dates"])
        == set(vx_proxy["session_dates"]),
        "no_dbn_files": not dbn_files,
        "no_cmbp_1_or_tcbbo": not forbidden,
    }
    quality = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(root),
        "window": {"start": "2025-08-01", "end": "2026-07-31"},
        "coverage_basis": "251 observed SPXW PM 0DTE option-market sessions",
        "option_schemas": option,
        "official_context": {"SPX": spx, "VIX": vix},
        "aligned": {"minute_entry": minute, "normalized": normalized},
        "futures_context": {"ES": es, "VX": vx, "SPX_proxy": es_proxy, "VIX_proxy": vx_proxy},
        "pairing": {
            "option_sessions": len(canonical),
            "spx_paired": len(canonical & set(spx["session_dates"])),
            "vix_paired": len(canonical & set(vix["session_dates"])),
            "minute_entry_paired": len(canonical & set(minute["session_dates"])),
        },
        "expected_vendor_boundaries": {
            "cbbo-1s": ">=2025-02-20 (before target window)",
            "VX": ">=2026-04-01",
            "ThetaData": ">=2024-10-01 (before target window)",
        },
        "vendor_quality_notices": {
            "Databento_OPRA_degraded_sessions": ["2025-10-22", "2026-07-31"],
            "Databento_futures_degraded_requested_sessions": [
                "2025-09-17",
                "2025-09-24",
                "2025-11-28",
                "2026-03-16",
                "2026-04-10",
            ],
            "Databento_XCBF_degraded_before_VX_requested_boundary": [
                "2025-09-30",
                "2025-10-01",
                "2026-01-16",
                "2026-01-21",
                "2026-02-18",
            ],
        },
        "dbn_files": dbn_files,
        "forbidden_schema_files": forbidden,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    (audit / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n"
    )

    futures_rows = _read_jsonl(audit / "futures_downloads.jsonl")
    unique_futures_rows = {
        str(row.get("raw_parquet_path")): row for row in futures_rows
    }
    successful_file_cost = sum(
        float(row.get("cost_estimate_usd", 0.0))
        for row in unique_futures_rows.values()
    )
    preflight_path = audit / "futures_cost_preflight.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}
    preflight_total = preflight.get("estimated_total_usd")
    interrupted_request_upper_bound = 0.001423805952
    conservative_spend = (
        float(preflight_total) + interrupted_request_upper_bound
        if preflight_total is not None
        else successful_file_cost + interrupted_request_upper_bound
    )
    spend = {
        "authorization_cap_usd": 9.0,
        "Databento_OPRA_subscription_spend_usd": 0.0,
        "ThetaData_existing_subscription_spend_usd": 0.0,
        "ES_VX_successful_file_cost_estimate_unique_usd": successful_file_cost,
        "ES_VX_preflight_estimated_total_usd": preflight_total,
        "interrupted_ES_request_upper_bound_usd": interrupted_request_upper_bound,
        "total_paid_spend_conservative_estimate_usd": conservative_spend,
        "actual_vendor_invoice_usd": (
            "UNKNOWN; bounded by cost preflight plus one interrupted ES request"
        ),
        "within_authorization": conservative_spend <= 9.0,
        "audit_records": len(futures_rows),
        "unique_successful_files": len(unique_futures_rows),
        "duplicate_resume_records_ignored": len(futures_rows) - len(unique_futures_rows),
        "excluded_paid_items": ["older-than-12-month L1", "cmbp-1", "tcbbo"],
    }
    (audit / "spend_ledger.json").write_text(json.dumps(spend, indent=2, sort_keys=True) + "\n")

    acquisition = {
        "corpus_root": str(root),
        "landing_policy": "outside repository and outside iCloud-synced Documents path",
        "window": {"start": "2025-08-01", "end": "2026-07-31"},
        "sources": {
            "Databento_OPRA": ["definition", "cbbo-1m", "ohlcv-1m", "statistics", "cbbo-1s"],
            "ThetaData": ["SPX index 1m", "VIX index 1m"],
            "Databento_futures_context": ["ES.c.0 ohlcv-1m RTH", "VX.c.0 ohlcv-1m RTH"],
        },
        "format": "Parquet-only raw/vendor corpus; pickle minute entry substrate; Parquet exit labels",
        "quality_report": str(audit / "data_quality_report.json"),
        "spend_ledger": str(audit / "spend_ledger.json"),
    }
    (audit / "acquisition_manifest.json").write_text(
        json.dumps(acquisition, indent=2, sort_keys=True) + "\n"
    )

    integrity = _write_integrity_manifest(root, audit / "integrity_sha256.jsonl")
    (audit / "integrity_summary.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "all_quality_checks_passed": quality["all_checks_passed"],
                "integrity": integrity,
                "spend": spend,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if quality["all_checks_passed"] and spend["within_authorization"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
