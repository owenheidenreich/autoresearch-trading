"""Validate the owned terminal SPX source for SPXW PM cash accounting.

The causal-day simulator must not turn a missing 16:00 option bid into an
invented fill.  It may, however, account for an expiring PM-settled contract at
intrinsic value when the underlying settlement source is independently
identified and validated.  This audit binds that source to the owned,
non-derived official SPX minute file and proves its identity with the aligned
option snapshot for every included episode.

No strategy outcome, threshold, or model is inspected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256


ET = "America/New_York"
DEFAULT_CORPUS_ROOT = Path(
    "/Users/och/.autoresearch-trading/pathd_2025-08-01_2026-07-31"
)
DEFAULT_COVERAGE_CSV = Path(
    "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/"
    "session_coverage.csv"
)

OFFICIAL_COLUMNS = (
    "event_time",
    "symbol",
    "close",
    "context_source",
    "is_derived",
    "is_proxy",
    "is_official_index_data",
)
QUOTE_COLUMNS = (
    "event_time",
    "expiry",
    "root",
    "settlement_style",
    "settlement_time_utc",
    "underlying_price",
)


class SettlementAuditError(RuntimeError):
    """The owned files do not prove a unique terminal settlement source."""


def _utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise SettlementAuditError("terminal source contains an invalid timestamp")
    return parsed


def validate_session(
    session: str,
    official: pd.DataFrame,
    quotes: pd.DataFrame,
) -> dict[str, Any]:
    """Return the unique locally-proven PM settlement value for one session."""

    missing_official = sorted(set(OFFICIAL_COLUMNS) - set(official.columns))
    missing_quotes = sorted(set(QUOTE_COLUMNS) - set(quotes.columns))
    if missing_official or missing_quotes:
        raise SettlementAuditError(
            f"{session}: source schema missing official={missing_official}, quotes={missing_quotes}"
        )
    if official.empty or quotes.empty:
        raise SettlementAuditError(f"{session}: terminal source is empty")

    official_time = _utc(official["event_time"])
    quote_time = _utc(quotes["event_time"])
    settlement_time = _utc(quotes["settlement_time_utc"])

    if not official["symbol"].astype(str).eq("SPX").all():
        raise SettlementAuditError(f"{session}: official source is not uniformly SPX")
    if not official["is_official_index_data"].eq(True).all():  # noqa: E712
        raise SettlementAuditError(f"{session}: official-index flag is not uniformly true")
    if not official["is_derived"].eq(False).all():  # noqa: E712
        raise SettlementAuditError(f"{session}: official source contains derived rows")
    if not official["is_proxy"].eq(False).all():  # noqa: E712
        raise SettlementAuditError(f"{session}: official source contains proxy rows")
    if not quotes["root"].astype(str).eq("SPXW").all():
        raise SettlementAuditError(f"{session}: option source contains a non-SPXW root")
    if not quotes["settlement_style"].astype(str).eq("PM").all():
        raise SettlementAuditError(f"{session}: option source contains a non-PM contract")
    expiry = pd.to_datetime(quotes["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    if not expiry.eq(session).all():
        raise SettlementAuditError(f"{session}: option source contains a non-same-day expiry")
    if settlement_time.nunique() != 1:
        raise SettlementAuditError(f"{session}: option definitions disagree on settlement time")

    declared_settlement = pd.Timestamp(settlement_time.iloc[0])
    if declared_settlement.tz_convert(ET).strftime("%Y-%m-%d %H:%M") != f"{session} 16:00":
        raise SettlementAuditError(f"{session}: PM settlement is not declared at 16:00 ET")

    final_time = pd.Timestamp(official_time.max())
    if final_time != declared_settlement:
        raise SettlementAuditError(
            f"{session}: official final row {final_time} != declared settlement {declared_settlement}"
        )
    final_rows = official.loc[official_time.eq(final_time)]
    final_close = pd.to_numeric(final_rows["close"], errors="coerce").dropna().unique()
    if len(final_close) != 1 or not np.isfinite(final_close[0]):
        raise SettlementAuditError(f"{session}: official final SPX close is not unique and finite")

    terminal_quotes = quotes.loc[quote_time.eq(declared_settlement)]
    aligned = pd.to_numeric(
        terminal_quotes["underlying_price"], errors="coerce"
    ).dropna().unique()
    if len(aligned) != 1:
        raise SettlementAuditError(f"{session}: aligned 16:00 SPX snapshot is not unique")
    if not np.isclose(float(aligned[0]), float(final_close[0]), atol=1e-12, rtol=0.0):
        raise SettlementAuditError(
            f"{session}: aligned terminal SPX {aligned[0]} != official close {final_close[0]}"
        )

    sources = official["context_source"].astype(str).unique()
    if len(sources) != 1:
        raise SettlementAuditError(f"{session}: official SPX provenance is not unique")

    return {
        "session": session,
        "settlement_spx": float(final_close[0]),
        "settlement_time_utc": declared_settlement.isoformat(),
        "settlement_time_et": declared_settlement.tz_convert(ET).isoformat(),
        "official_rows": int(len(official)),
        "terminal_option_rows": int(len(terminal_quotes)),
        "official_context_source": str(sources[0]),
        "aligned_underlying_spx": float(aligned[0]),
        "exact_aligned_identity": True,
        "official_non_derived_non_proxy": True,
        "same_day_spxw_pm_only": True,
    }


def run(
    corpus_root: Path,
    coverage_csv: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if evidence_dir.exists():
        raise SettlementAuditError(f"refusing to overwrite evidence: {evidence_dir}")
    coverage = pd.read_csv(coverage_csv)
    sessions = coverage.loc[
        coverage["included_for_episode_build"].astype(bool), "session"
    ].astype(str).tolist()
    if not sessions:
        raise SettlementAuditError("coverage contains no included sessions")

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for session in sessions:
        official_path = (
            corpus_root / "raw/index/spx_1m" / f"{session}.official_spx.parquet"
        )
        quote_path = (
            corpus_root
            / "aligned/normalized"
            / f"databento_spxw_0dte_{session}.parquet"
        )
        if not official_path.is_file() or not quote_path.is_file():
            raise SettlementAuditError(f"{session}: required terminal source file is missing")
        official = pd.read_parquet(official_path, columns=list(OFFICIAL_COLUMNS))
        quotes = pd.read_parquet(quote_path, columns=list(QUOTE_COLUMNS))
        record = validate_session(session, official, quotes)
        record["official_file"] = str(official_path)
        record["option_file"] = str(quote_path)
        records.append(record)
        for role, path in (("official_spx", official_path), ("aligned_options", quote_path)):
            sources.append(
                {
                    "session": session,
                    "role": role,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )

    evidence_dir.mkdir(parents=True, exist_ok=False)
    settlement_path = evidence_dir / "settlements.csv"
    manifest_path = evidence_dir / "source_manifest.json"
    pd.DataFrame(records).to_csv(settlement_path, index=False)
    manifest_path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")

    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-terminal-settlement-audit.v1",
        "created_on": "2026-08-14",
        "purpose": "validate the exact owned 16:00 SPX source used only for terminal PM cash accounting",
        "sessions": {
            "n": len(records),
            "first": min(sessions),
            "last": max(sessions),
            "all_exact_identity": True,
        },
        "source_contract": {
            "option_root": "SPXW",
            "option_expiry": "same session",
            "settlement_style": "PM",
            "declared_settlement_time": "16:00 America/New_York",
            "underlying": "SPX",
            "underlying_source": "owned official_spx minute parquet",
            "required_flags": {
                "is_official_index_data": True,
                "is_derived": False,
                "is_proxy": False,
            },
        },
        "validation": {
            "official_final_timestamp_equals_contract_settlement_timestamp": len(records),
            "aligned_terminal_underlying_equals_official_spx_close": len(records),
            "mismatches": 0,
            "status": "VALIDATED_FOR_TERMINAL_ACCOUNTING",
        },
        "accounting_law": {
            "call_intrinsic": "max(0, settlement_spx - strike)",
            "put_intrinsic": "max(0, strike - settlement_spx)",
            "execution_claim": False,
            "scope": "terminal cash accounting only after no executable bid remains through 16:00",
            "zero_recovery_sensitivity_still_required": True,
        },
        "important_limit": (
            "This is an owned official SPX close plus the normalized SPXW PM settlement timestamp, "
            "not an option bid and not a separately downloaded OCC exercise statement."
        ),
        "artifacts": {
            "settlements": {
                "path": str(settlement_path),
                "size_bytes": settlement_path.stat().st_size,
                "sha256": file_sha256(settlement_path),
            },
            "source_manifest": {
                "path": str(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": file_sha256(manifest_path),
            },
        },
        "model_fit": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path = evidence_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.corpus_root, args.coverage_csv, args.evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
