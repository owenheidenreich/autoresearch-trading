#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}
CRITICAL_ANOMALIES = {"metric_inconsistent", "trades_per_day_extreme"}


def _jsonl_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield lineno, json.loads(raw)
            except json.JSONDecodeError:
                continue


def _safe_iso(ts: Any) -> str:
    if not ts:
        return dt.datetime.now().isoformat()
    try:
        # Normalize "Z" if present.
        ts_s = str(ts).replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(ts_s)
        return parsed.isoformat()
    except Exception:
        return dt.datetime.now().isoformat()


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return default


def _incident_id(subsystem: str, severity: str, signature: str) -> str:
    payload = f"{subsystem}|{severity}|{signature}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _event(
    *,
    ts: Any,
    source: str,
    subsystem: str,
    severity: str,
    event_type: str,
    signature: str,
    ref_path: str,
    ref_line: int | None = None,
    run_id: str | None = None,
    experiment_id: int | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_time": _safe_iso(ts),
        "source": source,
        "subsystem": subsystem,
        "severity": severity,
        "event_type": event_type,
        "signature": signature,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "ref_path": ref_path,
        "ref_line": ref_line,
        "payload_json": json.dumps(payload or {}, sort_keys=True),
    }


def _incident_candidate(
    *, ts: Any, subsystem: str, severity: str, signature: str, ref_path: str
) -> dict[str, Any]:
    return {
        "ts": _safe_iso(ts),
        "subsystem": subsystem,
        "severity": severity,
        "signature": signature,
        "ref_path": ref_path,
    }


def _collect_experiments(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    exp_paths = sorted(results_root.glob("run-*/experiments.v2.jsonl"))

    for path in exp_paths:
        run_id = path.parent.name
        for lineno, rec in _jsonl_lines(path):
            ts = rec.get("timestamp")
            exp_id = rec.get("experiment_id")
            failure_type = str(rec.get("failure_type", "none"))
            anomaly_flags = rec.get("anomaly_flags") or []
            if not isinstance(anomaly_flags, list):
                anomaly_flags = []

            score_raw = rec.get("score")
            score = None
            try:
                score = float(score_raw) if score_raw is not None else None
            except Exception:
                score = None

            payload = {
                "kept": bool(rec.get("kept", False)),
                "score": score,
                "failure_type": failure_type,
                "anomaly_flags": anomaly_flags,
                "change_summary": rec.get("change_summary"),
                "file_name": path.name,
            }
            events.append(
                _event(
                    ts=ts,
                    source="inner_loop",
                    subsystem="training_loop",
                    severity="info",
                    event_type="experiment_result",
                    signature="experiment.result",
                    run_id=run_id,
                    experiment_id=exp_id,
                    ref_path=str(path),
                    ref_line=lineno,
                    payload=payload,
                )
            )

            if failure_type and failure_type != "none":
                sev = "warning"
                if failure_type in {"safety", "drift_guard", "train_crash", "timeout"}:
                    sev = "critical"
                incidents.append(
                    _incident_candidate(
                        ts=ts,
                        subsystem="training_loop",
                        severity=sev,
                        signature=f"exp.failure_type:{failure_type}",
                        ref_path=f"{path}#L{lineno}",
                    )
                )

            for flag in anomaly_flags:
                sev = "critical" if flag in CRITICAL_ANOMALIES else "warning"
                incidents.append(
                    _incident_candidate(
                        ts=ts,
                        subsystem="training_loop",
                        severity=sev,
                        signature=f"exp.anomaly:{flag}",
                        ref_path=f"{path}#L{lineno}",
                    )
                )
    return events, incidents


def _collect_live_audit(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    audit_path = results_root / "live" / "audit.jsonl"
    if not audit_path.exists():
        return events, incidents

    for lineno, rec in _jsonl_lines(audit_path):
        ts = rec.get("ts")
        event_name = str(rec.get("event", "unknown"))
        payload = rec.get("payload") or {}
        severity = "info"
        if isinstance(payload, dict):
            if payload.get("passed") is False and event_name == "entitlement_probe":
                severity = "critical"
            elif payload.get("passed") is False:
                severity = "warning"
            elif payload.get("error"):
                severity = "critical"
            elif payload.get("warnings"):
                severity = "warning"
        if "fail" in event_name.lower() or "error" in event_name.lower():
            severity = "critical"

        signature = f"live.{event_name}"
        events.append(
            _event(
                ts=ts,
                source="outer_loop",
                subsystem="live",
                severity=severity,
                event_type="live_audit",
                signature=signature,
                ref_path=str(audit_path),
                ref_line=lineno,
                payload=payload if isinstance(payload, dict) else {"raw_payload": payload},
            )
        )
        if severity != "info":
            incidents.append(
                _incident_candidate(
                    ts=ts,
                    subsystem="live",
                    severity=severity,
                    signature=f"{signature}.issue",
                    ref_path=f"{audit_path}#L{lineno}",
                )
            )
    return events, incidents


def _collect_replay(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []

    for journal in sorted(results_root.rglob("*_journal.json")):
        try:
            rec = json.loads(journal.read_text())
        except Exception:
            continue

        replay_date = rec.get("replay_date")
        trades = rec.get("trades") or []
        bar_log = rec.get("bar_log") or []
        session_stats = rec.get("session_stats") or {}
        journal_ts = rec.get("generated_at") or rec.get("created_at") or dt.datetime.now().isoformat()
        events.append(
            _event(
                ts=journal_ts,
                source="outer_loop",
                subsystem="replay",
                severity="info",
                event_type="replay_journal",
                signature="replay.journal",
                ref_path=str(journal),
                payload={
                    "replay_date": replay_date,
                    "num_trades": len(trades),
                    "total_bars": session_stats.get("total_bars", len(bar_log)),
                    "avg_gate_prob": session_stats.get("avg_gate_prob"),
                    "cooldown_blocked": session_stats.get("cooldown_blocked"),
                    "pre_10am_blocked": session_stats.get("pre_10am_blocked"),
                },
            )
        )
        if len(trades) == 0:
            incidents.append(
                _incident_candidate(
                    ts=journal_ts,
                    subsystem="replay",
                    severity="warning",
                    signature="replay.no_trades",
                    ref_path=str(journal),
                )
            )

    for qa_path in sorted(results_root.rglob("*_qa.json")):
        try:
            qa = json.loads(qa_path.read_text())
        except Exception:
            continue
        ts = dt.datetime.now().isoformat()
        passed = bool(qa.get("passed", False))
        critical_count = int(qa.get("critical_count", 0) or 0)
        warning_count = int(qa.get("warning_count", 0) or 0)
        severity = "info"
        if not passed or critical_count > 0:
            severity = "critical"
        elif warning_count > 0:
            severity = "warning"
        events.append(
            _event(
                ts=ts,
                source="outer_loop",
                subsystem="replay",
                severity=severity,
                event_type="replay_qa",
                signature="replay.qa",
                ref_path=str(qa_path),
                payload={
                    "passed": passed,
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                    "anomalies": qa.get("anomalies", []),
                },
            )
        )
        if not passed:
            incidents.append(
                _incident_candidate(
                    ts=ts,
                    subsystem="replay",
                    severity="critical",
                    signature="replay.qa_failed",
                    ref_path=str(qa_path),
                )
            )
        for anomaly in qa.get("anomalies", []) if isinstance(qa.get("anomalies"), list) else []:
            if not isinstance(anomaly, dict):
                continue
            code = str(anomaly.get("code", "unknown"))
            sev = str(anomaly.get("severity", "warning")).lower()
            sev_norm = "critical" if sev == "critical" else "warning"
            incidents.append(
                _incident_candidate(
                    ts=ts,
                    subsystem="replay",
                    severity=sev_norm,
                    signature=f"replay.qa.{code}",
                    ref_path=str(qa_path),
                )
            )

    for day_csv in sorted(results_root.rglob("*_ledger_days.csv")):
        try:
            df_day = pd.read_csv(day_csv)
        except Exception:
            continue
        if df_day.empty:
            continue
        row = df_day.iloc[0].to_dict()
        ts = dt.datetime.now().isoformat()
        qa_passed = _as_bool(row.get("qa_passed"), default=True)
        sev = "info" if qa_passed else "warning"
        events.append(
            _event(
                ts=ts,
                source="outer_loop",
                subsystem="replay",
                severity=sev,
                event_type="replay_day_ledger",
                signature="replay.day_ledger",
                ref_path=str(day_csv),
                payload={
                    "replay_date": row.get("replay_date"),
                    "num_trades": int(row.get("num_trades", 0) or 0),
                    "total_bars": int(row.get("total_bars", 0) or 0),
                    "total_pnl_pct": float(row.get("total_pnl_pct", 0.0) or 0.0),
                    "qa_passed": qa_passed,
                    "qa_critical_count": int(row.get("qa_critical_count", 0) or 0),
                    "qa_warning_count": int(row.get("qa_warning_count", 0) or 0),
                },
            )
        )
        if not qa_passed:
            incidents.append(
                _incident_candidate(
                    ts=ts,
                    subsystem="replay",
                    severity="warning",
                    signature="replay.day_qa_failed",
                    ref_path=str(day_csv),
                )
            )

    # Also ingest canonical replay CSV trade logs (avoid training trade_log.csv files).
    for csv_path in sorted(results_root.rglob("*.csv")):
        name = csv_path.name.lower()
        parent = csv_path.parent.name.lower()
        if not (
            name.startswith("replay-")
            or name.endswith("_replay.csv")
            or parent.startswith("nightly-replay")
            or "replay" in parent
        ):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        events.append(
            _event(
                ts=dt.datetime.now().isoformat(),
                source="outer_loop",
                subsystem="replay",
                severity="info",
                event_type="replay_trade_log",
                signature="replay.trade_log",
                ref_path=str(csv_path),
                payload={"rows": int(len(df))},
            )
        )
        if len(df) == 0:
            incidents.append(
                _incident_candidate(
                    ts=dt.datetime.now().isoformat(),
                    subsystem="replay",
                    severity="warning",
                    signature="replay.empty_trade_log",
                    ref_path=str(csv_path),
                )
            )

    return events, incidents


def _collect_loop_logs(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    for run_dir in sorted(results_root.glob("run-*")):
        path = run_dir / "loop.log"
        if not path.exists():
            continue
        run_id = run_dir.name
        run_date = run_id[4:14] if run_id.startswith("run-") and len(run_id) >= 14 else dt.date.today().isoformat()
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            s = line.strip()
            if not s:
                continue
            ts = dt.datetime.now().isoformat()
            # Parse [HH:MM:SS] lines and anchor to run date when available.
            if s.startswith("[") and "]" in s and len(s) > 10:
                hhmmss = s[1 : s.find("]")]
                try:
                    ts = dt.datetime.fromisoformat(f"{run_date}T{hhmmss}").isoformat()
                except Exception:
                    ts = dt.datetime.now().isoformat()

            sev = "info"
            sig = "loop.log.info"
            lower = s.lower()
            if " safety:" in lower or " drift_guard" in lower:
                sev = "critical"
                sig = "loop.log.contract_or_safety"
            elif "failed" in lower or "error" in lower:
                sev = "warning"
                sig = "loop.log.failure"

            events.append(
                _event(
                    ts=ts,
                    source="inner_loop",
                    subsystem="training_loop",
                    severity=sev,
                    event_type="loop_log",
                    signature=sig,
                    run_id=run_id,
                    ref_path=str(path),
                    ref_line=lineno,
                    payload={"line": s[:500]},
                )
            )
            if sev != "info":
                incidents.append(
                    _incident_candidate(
                        ts=ts,
                        subsystem="training_loop",
                        severity=sev,
                        signature=f"{sig}.issue",
                        ref_path=f"{path}#L{lineno}",
                    )
                )
    return events, incidents


def _build_incidents(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for c in candidates:
        key = (c["subsystem"], c["severity"], c["signature"])
        grouped.setdefault(key, []).append(c)

    rows: list[dict[str, Any]] = []
    for (subsystem, severity, signature), bucket in grouped.items():
        bucket_sorted = sorted(bucket, key=lambda x: x["ts"])
        rows.append(
            {
                "id": _incident_id(subsystem, severity, signature),
                "subsystem": subsystem,
                "severity": severity,
                "signature": signature,
                "first_seen": bucket_sorted[0]["ts"],
                "last_seen": bucket_sorted[-1]["ts"],
                "count": len(bucket_sorted),
                "latest_ref": bucket_sorted[-1]["ref_path"],
            }
        )
    rows.sort(key=lambda x: (-SEVERITY_RANK.get(x["severity"], 0), -x["count"], x["signature"]))
    return rows


def _write_incidents(path: Path, incidents: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in incidents:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _build_digest(
    *,
    digest_path: Path,
    total_events: int,
    incidents: list[dict[str, Any]],
    evidence_df: pd.DataFrame,
) -> None:
    critical = [i for i in incidents if i["severity"] == "critical"]
    warning = [i for i in incidents if i["severity"] == "warning"]

    exp_df = evidence_df[evidence_df["event_type"] == "experiment_result"].copy()
    regressions: list[str] = []
    if not exp_df.empty:
        scores = []
        for _, row in exp_df.sort_values("event_time").iterrows():
            payload = json.loads(row["payload_json"])
            score = payload.get("score")
            if isinstance(score, (float, int)):
                scores.append(float(score))
        if scores:
            peak = max(scores)
            trough = min(scores)
            regressions.append(
                f"- Score range observed: min={trough:.4f}, max={peak:.4f}, spread={(peak - trough):.4f}"
            )

    top_incidents = incidents[:8]
    recommendations: list[str] = []
    sigs = {i["signature"] for i in incidents}
    if "exp.failure_type:drift_guard" in sigs:
        recommendations.append(
            "- Add explicit contract assertions to `program.md` for any newly allowed behavior before rerunning."
        )
    if any(s.startswith("exp.anomaly:do_nothing_zero") for s in sigs):
        recommendations.append(
            "- Add a pre-keep hard gate for `do_nothing_pct` lower bound and verify action decoding in eval."
        )
    if any(s.startswith("live.entitlement_probe") for s in sigs):
        recommendations.append(
            "- Treat live entitlement failures as release blockers before enabling live paper execution."
        )
    if not recommendations:
        recommendations.append("- Run one controlled near-tie experiment to validate stability guard bands.")

    digest_lines = [
        f"# Decision Digest ({dt.date.today().isoformat()})",
        "",
        "## Snapshot",
        f"- Total evidence events: {total_events}",
        f"- Incidents: {len(incidents)} (critical={len(critical)}, warning={len(warning)})",
        "",
        "## Top Regressions",
    ]
    if regressions:
        digest_lines.extend(regressions)
    else:
        digest_lines.append("- No score regression evidence available yet.")

    digest_lines.extend(["", "## Broken Assumptions"])
    if top_incidents:
        for inc in top_incidents:
            digest_lines.append(
                f"- [{inc['severity']}] `{inc['signature']}` x{inc['count']} "
                f"(first={inc['first_seen']}, last={inc['last_seen']})"
            )
    else:
        digest_lines.append("- No incidents detected.")

    digest_lines.extend(["", "## Recommended Next Experiments"])
    digest_lines.extend(recommendations)
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text("\n".join(digest_lines) + "\n")


def ingest(results_root: Path, output_root: Path) -> dict[str, Any]:
    exp_events, exp_incidents = _collect_experiments(results_root)
    replay_events, replay_incidents = _collect_replay(results_root)
    live_events, live_incidents = _collect_live_audit(results_root)
    loop_events, loop_incidents = _collect_loop_logs(results_root)

    events = exp_events + replay_events + live_events + loop_events
    incidents = _build_incidents(exp_incidents + replay_incidents + live_incidents + loop_incidents)
    output_root.mkdir(parents=True, exist_ok=True)

    evidence_path = output_root / "evidence.parquet"
    incidents_path = output_root / "incidents.jsonl"
    digest_path = output_root / f"decision-digest-{dt.date.today().isoformat()}.md"

    if events:
        df = pd.DataFrame(events).sort_values("event_time")
    else:
        df = pd.DataFrame(
            columns=[
                "event_time",
                "source",
                "subsystem",
                "severity",
                "event_type",
                "signature",
                "run_id",
                "experiment_id",
                "ref_path",
                "ref_line",
                "payload_json",
            ]
        )
    df.to_parquet(evidence_path, index=False)
    _write_incidents(incidents_path, incidents)
    _build_digest(
        digest_path=digest_path,
        total_events=len(df),
        incidents=incidents,
        evidence_df=df,
    )

    return {
        "events": len(df),
        "incidents": len(incidents),
        "evidence_path": str(evidence_path),
        "incidents_path": str(incidents_path),
        "digest_path": str(digest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest inner/outer loop evidence into analysis artifacts")
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Path to results root (default: ./results)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/analysis",
        help="Path for analysis outputs (default: results/analysis)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    summary = ingest(results_root, output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
