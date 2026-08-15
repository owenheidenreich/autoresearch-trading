"""Protocol163 recent-data integrity audit.

This audit is deliberately scoped to the April-May 2026 catch-up block. The
older generic sufficiency audit is tied to the first pilot windows, while this
one answers the narrower question: is the newly downloaded official-context
block clean enough to feed the serial one-account Protocol163 training path?
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLOSED_SESSIONS = {"2026-04-03"}
EXPECTED_DECISION_ROWS = 360
REQUIRED_SCHEMAS = ("definition", "cbbo-1m", "ohlcv-1m", "statistics")
CONTRACT_RE = re.compile(r"^SPXW-\d{8}-(\d+\.\d+)-[CP]$")


@dataclass(frozen=True)
class SessionAudit:
    session: str
    neural_rows: int
    missing_decision_minutes: list[str]
    duplicate_decision_minutes: int
    normalized_rows: int
    invalid_roots: int
    invalid_settlement: int
    invalid_strike_alignment: int
    malformed_quote_rows: int
    malformed_quote_fraction: float
    candidate_count: int
    bad_candidate_quotes: int
    bad_candidate_greeks: int
    bad_candidate_contract_ids: int
    oi_stat9_rows: int
    oi_mismatches: int
    spx_official_fraction: float
    vix_official_fraction: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=Path("v4/audit/protocol163_recent_neural_build_summary.json"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_data_integrity"))
    return p.parse_args()


def expected_sessions(start: str, end: str) -> list[str]:
    cursor = pd.Timestamp(start).date()
    final = pd.Timestamp(end).date()
    out: list[str] = []
    while cursor <= final:
        label = cursor.isoformat()
        if cursor.weekday() < 5 and label not in CLOSED_SESSIONS:
            out.append(label)
        cursor += timedelta(days=1)
    return out


def missing_minutes(decision_times: list[pd.Timestamp]) -> list[str]:
    if not decision_times:
        return []
    seen = set(decision_times)
    cur = min(decision_times)
    end = max(decision_times)
    missing: list[str] = []
    while cur <= end:
        if cur not in seen:
            missing.append(cur.isoformat())
        cur += pd.Timedelta(minutes=1)
    return missing


def _read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(path), columns=columns)


def _official_fraction(path: str | Path) -> float:
    frame = _read_parquet(path)
    if frame.empty or "is_official_index_data" not in frame.columns:
        return 0.0
    return float(frame["is_official_index_data"].astype(bool).mean())


def _stat9_lookup(raw_root: Path, session: str) -> tuple[dict[str, int], int]:
    path = raw_root / "databento" / "opra_spxw_statistics" / f"{session}.statistics.parquet"
    if not path.exists():
        return {}, 0
    stats = _read_parquet(path, columns=["symbol", "stat_type", "quantity"])
    stat9 = stats[pd.to_numeric(stats["stat_type"], errors="coerce") == 9].copy()
    lookup: dict[str, int] = {}
    for row in stat9.to_dict("records"):
        symbol = str(row["symbol"]).strip()
        quantity = row.get("quantity")
        if pd.notna(quantity):
            lookup[symbol] = int(quantity)
    return lookup, int(len(stat9))


def _audit_normalized(record: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    cols = ["root", "settlement_style", "strike", "bid", "ask", "raw_symbol", "stat_open_interest"]
    frame = _read_parquet(record["derived_normalized_path"], columns=cols)
    strike = pd.to_numeric(frame["strike"], errors="coerce")
    bid = pd.to_numeric(frame["bid"], errors="coerce")
    ask = pd.to_numeric(frame["ask"], errors="coerce")
    malformed = frame[bid.isna() | ask.isna() | (bid < 0) | (ask <= 0) | (ask < bid)]

    lookup, stat9_rows = _stat9_lookup(raw_root, record["session"])
    oi_mismatches = 0
    if lookup:
        oi = frame[["raw_symbol", "stat_open_interest"]].dropna().drop_duplicates("raw_symbol")
        for row in oi.to_dict("records"):
            raw_symbol = str(row["raw_symbol"]).strip()
            expected = lookup.get(raw_symbol)
            if expected is not None and int(row["stat_open_interest"]) != expected:
                oi_mismatches += 1

    return {
        "normalized_rows": int(len(frame)),
        "invalid_roots": int((frame["root"].astype(str) != "SPXW").sum()),
        "invalid_settlement": int((frame["settlement_style"].astype(str) != "PM").sum()),
        "invalid_strike_alignment": int((~np.isclose(strike % 5.0, 0.0)).sum()),
        "malformed_quote_rows": int(len(malformed)),
        "malformed_quote_fraction": float(len(malformed) / len(frame)) if len(frame) else 0.0,
        "oi_stat9_rows": stat9_rows,
        "oi_mismatches": int(oi_mismatches),
    }


def _audit_neural(record: dict[str, Any]) -> dict[str, Any]:
    rows = pickle.loads(Path(record["neural_path"]).read_bytes())
    decision_times = [pd.Timestamp(row["decision_time"]).tz_convert("UTC") for row in rows]
    duplicate_count = len(decision_times) - len(set(decision_times))
    missing = missing_minutes(decision_times)

    candidate_count = 0
    bad_quotes = 0
    bad_greeks = 0
    bad_contract_ids = 0
    for row in rows:
        names = {name: idx for idx, name in enumerate(row["feature_names"])}
        ladder = row["option_ladder"]
        mask = row["candidate_mask"].astype(bool)
        contracts = row["contract_ids"]
        if not mask.any():
            continue
        bid = ladder[:, :, names["bid"]][mask]
        ask = ladder[:, :, names["ask"]][mask]
        mid = ladder[:, :, names["mid"]][mask]
        candidate_count += int(mask.sum())
        bad_quotes += int((~np.isfinite(bid) | ~np.isfinite(ask) | ~np.isfinite(mid) | (bid < 0) | (ask <= 0) | (ask < bid)).sum())
        for greek_name in ("iv", "delta", "gamma", "theta"):
            values = ladder[:, :, names[greek_name]][mask]
            bad_greeks += int((~np.isfinite(values)).sum())
        for contract_id in contracts[mask]:
            text = str(contract_id)
            match = CONTRACT_RE.match(text)
            if match is None or float(match.group(1)) % 5.0 != 0.0:
                bad_contract_ids += 1

    return {
        "neural_rows": int(len(rows)),
        "missing_decision_minutes": missing,
        "duplicate_decision_minutes": int(duplicate_count),
        "candidate_count": int(candidate_count),
        "bad_candidate_quotes": int(bad_quotes),
        "bad_candidate_greeks": int(bad_greeks),
        "bad_candidate_contract_ids": int(bad_contract_ids),
    }


def _download_audit_spend(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if line.strip():
            total += float(json.loads(line).get("cost_estimate_usd", 0.0) or 0.0)
    return total


def run(summary_path: Path, raw_root: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    records = {record["session"]: record for record in summary["records"]}
    expected = expected_sessions(summary["start_date"], summary["end_date"])
    missing_built = [session for session in expected if session not in records]
    unexpected_skipped = [session for session in summary.get("sessions_skipped", []) if session not in CLOSED_SESSIONS]

    audits: list[SessionAudit] = []
    for session in expected:
        record = records.get(session)
        if record is None:
            continue
        normalized = _audit_normalized(record, raw_root)
        neural = _audit_neural(record)
        audits.append(
            SessionAudit(
                session=session,
                spx_official_fraction=_official_fraction(record["spx_path"]),
                vix_official_fraction=_official_fraction(record["vix_path"]),
                **normalized,
                **neural,
            )
        )

    hard_failures: list[str] = []
    warnings: list[str] = []
    if missing_built:
        hard_failures.append(f"missing built sessions: {missing_built}")
    if unexpected_skipped:
        hard_failures.append(f"unexpected skipped sessions: {unexpected_skipped}")

    for audit in audits:
        if audit.invalid_roots:
            hard_failures.append(f"{audit.session}: non-SPXW rows={audit.invalid_roots}")
        if audit.invalid_settlement:
            hard_failures.append(f"{audit.session}: non-PM settlement rows={audit.invalid_settlement}")
        if audit.invalid_strike_alignment:
            hard_failures.append(f"{audit.session}: non-$5 strike rows={audit.invalid_strike_alignment}")
        if audit.oi_mismatches:
            hard_failures.append(f"{audit.session}: OI mismatches vs stat_type 9={audit.oi_mismatches}")
        if audit.bad_candidate_quotes:
            hard_failures.append(f"{audit.session}: bad tradable candidate quotes={audit.bad_candidate_quotes}")
        if audit.bad_candidate_greeks:
            hard_failures.append(f"{audit.session}: bad tradable candidate Greeks={audit.bad_candidate_greeks}")
        if audit.bad_candidate_contract_ids:
            hard_failures.append(f"{audit.session}: bad tradable candidate ids={audit.bad_candidate_contract_ids}")
        if audit.spx_official_fraction < 1.0 or audit.vix_official_fraction < 1.0:
            hard_failures.append(
                f"{audit.session}: official context fractions spx={audit.spx_official_fraction:.3f} "
                f"vix={audit.vix_official_fraction:.3f}"
            )
        if audit.neural_rows != EXPECTED_DECISION_ROWS:
            warnings.append(
                f"{audit.session}: neural_rows={audit.neural_rows}, "
                f"missing_minutes={audit.missing_decision_minutes}"
            )
        if audit.malformed_quote_fraction > 0.25:
            warnings.append(
                f"{audit.session}: high raw quote malformed fraction={audit.malformed_quote_fraction:.3f}; "
                "candidate mask still passed executable quote checks"
            )

    payload = {
        "decision": "ready_for_protocol163_training" if not hard_failures else "blocked_data_integrity_failure",
        "summary_path": str(summary_path),
        "expected_sessions": expected,
        "closed_sessions": sorted(CLOSED_SESSIONS),
        "missing_built_sessions": missing_built,
        "unexpected_skipped_sessions": unexpected_skipped,
        "sessions_audited": len(audits),
        "total_neural_rows": sum(a.neural_rows for a in audits),
        "total_candidates": sum(a.candidate_count for a in audits),
        "databento_estimated_spend_usd": round(
            _download_audit_spend(Path("v4/audit/databento_protocol163_recent_catchup_downloads.jsonl")),
            4,
        ),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "sessions": [asdict(audit) for audit in audits],
    }
    return payload


def _write_report(payload: dict[str, Any], out_dir: Path) -> Path:
    report = out_dir / "report.md"
    lines = [
        "# Protocol163 Recent Data Integrity",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        f"- Sessions audited: {payload['sessions_audited']}",
        f"- Total neural rows: {payload['total_neural_rows']}",
        f"- Total tradable candidates: {payload['total_candidates']}",
        f"- Databento estimated spend: ${payload['databento_estimated_spend_usd']:.4f}",
        f"- Closed sessions skipped: {', '.join(payload['closed_sessions'])}",
        "",
        "## Hard Failures",
        "",
    ]
    if payload["hard_failures"]:
        lines.extend(f"- {item}" for item in payload["hard_failures"])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if payload["warnings"]:
        lines.extend(f"- {item}" for item in payload["warnings"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Primary training rows are official-context rows; the broad context-provenance audit over the shared index directory can show older derived/proxy files, so this report checks only the Protocol163 build records.",
            "- Raw normalized quote rows may contain missing or wide quotes. That is expected in OPRA data; the hard check is that tradable candidates used by the model have executable bid/ask and finite Greeks.",
            "- A missing neural decision minute means no candidate survived the tradability/value filters at that minute; it is equivalent to a forced wait for entry training, but it remains visible here.",
        ]
    )
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    args = parse_args()
    payload = run(args.summary, args.raw_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path = _write_report(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "summary": str(summary_path), "report": str(report_path)}, indent=2))
    return 0 if payload["decision"] == "ready_for_protocol163_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
