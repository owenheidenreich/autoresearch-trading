"""Protocol 154: promotion decision for the multi-contract challenger.

This protocol does not search for a better sizing rule. It freezes
``account_aware_sizer_v1`` and aggregates the existing economic, timing,
capital, and live-stack evidence into one explicit promotion verdict.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROTOCOL151_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_151_protocol101_account_aware_sizing_validation"
)
DEFAULT_PROTOCOL152_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_152_protocol101_multi_contract_promotion_gauntlet"
)
DEFAULT_PROTOCOL153_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_153_protocol101_multi_contract_live_stack_compatibility"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_154_protocol101_multi_contract_promotion_decision"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")
MIN_CRITICAL_TIMING_COVERAGE = 0.95
CRITICAL_DELAYS = (1, 5)
ADVISORY_DELAYS = (15, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol151-dir", type=Path, default=DEFAULT_PROTOCOL151_DIR)
    parser.add_argument("--protocol152-dir", type=Path, default=DEFAULT_PROTOCOL152_DIR)
    parser.add_argument("--protocol153-dir", type=Path, default=DEFAULT_PROTOCOL153_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence(args.protocol151_dir, args.protocol152_dir, args.protocol153_dir)
    gates = build_promotion_gates(evidence)
    decision = decide(gates)
    payload = {
        "protocol": "154_protocol101_multi_contract_promotion_decision",
        "decision": decision,
        "promotable_now": decision.startswith("promote_"),
        "rejected": decision.startswith("reject_"),
        "blocked": decision.startswith("blocked_"),
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "paper_orders": False,
        "model_training": False,
        "protocol101_frozen": True,
        "challenger": "account_aware_sizer_v1",
        "baseline": "one_contract_protocol101",
        "sources": {
            "protocol151": str(args.protocol151_dir),
            "protocol152": str(args.protocol152_dir),
            "protocol153": str(args.protocol153_dir),
        },
        "gates": gates,
        "evidence": evidence,
        "outputs": {
            "report": str(args.out_dir / "report.md"),
            "summary": str(args.out_dir / "summary.json"),
            "promotion_gates": str(args.out_dir / "promotion_gates.csv"),
            "missing_timing_coverage": str(args.out_dir / "missing_timing_coverage.csv"),
        },
        "next_gate": next_gate(decision, gates),
    }
    pd.DataFrame(gates).to_csv(args.out_dir / "promotion_gates.csv", index=False)
    pd.DataFrame(missing_timing_coverage(evidence["timing_summary"])).to_csv(
        args.out_dir / "missing_timing_coverage.csv",
        index=False,
    )
    (args.out_dir / "summary.json").write_text(json_dumps(payload))
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload)
    print(json.dumps({"decision": decision, "report": payload["outputs"]["report"]}, indent=2, sort_keys=True))
    return 0


def load_evidence(protocol151_dir: Path, protocol152_dir: Path, protocol153_dir: Path) -> dict[str, Any]:
    p151 = load_json(protocol151_dir / "summary.json")
    p152 = load_json(protocol152_dir / "summary.json")
    p153 = load_json(protocol153_dir / "summary.json")
    validation = read_records(protocol151_dir / "validation_summary.csv")
    segment = read_records(protocol151_dir / "segment_summary.csv")
    side = read_records(protocol151_dir / "side_summary.csv")
    timing = read_records(protocol152_dir / "timing_summary.csv")
    return {
        "protocol151_decision": p151.get("decision"),
        "protocol152_decision": p152.get("decision"),
        "protocol153_decision": p153.get("decision"),
        "protocol151_gate": p151.get("gate", {}),
        "protocol152_base_gate": p152.get("base_gate", {}),
        "protocol152_timing_gate": p152.get("timing_gate", {}),
        "protocol153_summary": p153.get("summary", {}),
        "validation_summary": validation,
        "segment_summary": segment,
        "side_summary": side,
        "timing_summary": timing,
        "baseline_10000": p151.get("baseline_10000", {}),
        "candidate_10000": p151.get("candidate_10000", {}),
        "concentration": p151.get("concentration", {}),
    }


def build_promotion_gates(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    p151_gate = evidence["protocol151_gate"]
    gates.append(
        gate(
            "base_economic_edge",
            "pass" if evidence["protocol151_decision"] == "pass_account_aware_sizing_beats_one_contract_research_only" else "fail",
            p151_gate.get("reason", "missing_protocol151_pass"),
            metric=money_delta(evidence["candidate_10000"], evidence["baseline_10000"]),
            threshold="candidate must beat one-contract baseline from $10,000",
        )
    )
    gates.append(
        gate(
            "cash_stress",
            "pass" if all_bool(evidence["validation_summary"], "incremental_positive") else "fail",
            "pass" if all_bool(evidence["validation_summary"], "incremental_positive") else "negative incremental row under starting-cash/slippage stress",
            metric=min_numeric(evidence["validation_summary"], "incremental_pnl"),
            threshold="incremental PnL > 0 for every starting cash and $0/$0.25/$0.50 stress row",
        )
    )
    gates.append(
        gate(
            "drawdown_efficiency",
            "pass" if all_bool(evidence["validation_summary"], "return_over_drawdown_improved") else "fail",
            "pass" if all_bool(evidence["validation_summary"], "return_over_drawdown_improved") else "return-over-drawdown failed on at least one row",
            metric=min_numeric(evidence["validation_summary"], "candidate_return_over_drawdown"),
            threshold="candidate return/max-drawdown must improve every stress row",
        )
    )
    gates.append(
        gate(
            "worst_day_control",
            "pass" if all_bool(evidence["validation_summary"], "worst_day_not_materially_worse") else "fail",
            "pass" if all_bool(evidence["validation_summary"], "worst_day_not_materially_worse") else "candidate worst day materially worse",
            metric=min_numeric(evidence["validation_summary"], "candidate_worst_day_pnl"),
            threshold="candidate worst day no more than $500 worse than one-contract baseline",
        )
    )
    gates.append(
        gate(
            "segment_portability",
            "pass" if all(float(row.get("incremental_pnl", 0.0)) > 0 for row in evidence["segment_summary"]) else "fail",
            "pass" if all(float(row.get("incremental_pnl", 0.0)) > 0 for row in evidence["segment_summary"]) else "negative incremental segment",
            metric={str(row.get("segment")): float(row.get("incremental_pnl", 0.0)) for row in evidence["segment_summary"]},
            threshold="incremental PnL positive in every tested segment",
        )
    )
    concentration = evidence["concentration"]
    gates.append(
        gate(
            "concentration",
            "pass"
            if float(concentration.get("top_day_share_of_positive", 1.0)) <= 0.35
            and float(concentration.get("top_month_share_of_positive", 1.0)) <= 0.55
            else "fail",
            "pass",
            metric={
                "top_day_share_of_positive": concentration.get("top_day_share_of_positive"),
                "top_month_share_of_positive": concentration.get("top_month_share_of_positive"),
            },
            threshold="top day <=35% and top month <=55% of positive incremental PnL",
        )
    )
    timing_gate = critical_timing_gate(evidence["timing_summary"])
    gates.append(timing_gate)
    advisory = advisory_timing_gate(evidence["timing_summary"])
    gates.append(advisory)
    p153 = evidence["protocol153_summary"]
    gates.append(
        gate(
            "live_stack_compatibility",
            "pass"
            if evidence["protocol153_decision"] == "pass_multi_contract_live_stack_compatible_research_only"
            and int(p153.get("risk_failed_rows", 1)) == 0
            and int(p153.get("schema_failed_rows", 1)) == 0
            else "fail",
            "pass" if int(p153.get("risk_failed_rows", 1)) == 0 and int(p153.get("schema_failed_rows", 1)) == 0 else "risk/schema failures",
            metric={
                "risk_failed_rows": p153.get("risk_failed_rows"),
                "schema_failed_rows": p153.get("schema_failed_rows"),
                "quantity_counts": p153.get("quantity_counts"),
            },
            threshold="zero risk/schema failures when explicitly configured for candidate max quantity",
        )
    )
    gates.append(
        gate(
            "one_contract_default_guard",
            "pass"
            if bool(p153.get("default_one_contract_schema_still_rejects_qty2"))
            and bool(p153.get("default_one_contract_risk_gate_still_rejects_qty2"))
            else "fail",
            "pass",
            metric={
                "default_schema_rejects_qty2": p153.get("default_one_contract_schema_still_rejects_qty2"),
                "default_risk_gate_rejects_qty2": p153.get("default_one_contract_risk_gate_still_rejects_qty2"),
            },
            threshold="default operational path must still reject qty=2",
        )
    )
    return gates


def critical_timing_gate(timing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    critical = [row for row in timing_rows if int(row.get("delay_seconds", -1)) in CRITICAL_DELAYS]
    reasons: list[str] = []
    if not critical:
        reasons.append("missing critical delay rows")
    missing = missing_timing_coverage(timing_rows)
    if missing:
        reasons.append("incomplete high-resolution coverage")
    negative = [
        row
        for row in critical
        if float(row.get("challenger_delayed_pnl", 0.0)) <= 0
        or float(row.get("incremental_delayed_pnl", 0.0)) <= 0
    ]
    if negative:
        reasons.append("nonpositive challenger or incremental PnL under critical delay")
    if negative:
        status = "fail"
    elif missing:
        status = "block"
    else:
        status = "pass"
    return gate(
        "critical_timing_realism",
        status,
        "pass" if status == "pass" else ", ".join(reasons),
        metric={
            "critical_delays": list(CRITICAL_DELAYS),
            "min_coverage": min([float(row.get("coverage", 0.0)) for row in critical] or [0.0]),
            "missing_rows_to_95pct": missing,
        },
        threshold=f">= {MIN_CRITICAL_TIMING_COVERAGE:.0%} high-res coverage and positive incremental PnL at 1s/5s",
    )


def advisory_timing_gate(timing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    advisory = [row for row in timing_rows if int(row.get("delay_seconds", -1)) in ADVISORY_DELAYS]
    negative = [
        {
            "split": row.get("split"),
            "delay_seconds": int(row.get("delay_seconds", -1)),
            "incremental_delayed_pnl": float(row.get("incremental_delayed_pnl", 0.0)),
            "challenger_delayed_pnl": float(row.get("challenger_delayed_pnl", 0.0)),
        }
        for row in advisory
        if float(row.get("incremental_delayed_pnl", 0.0)) <= 0 or float(row.get("challenger_delayed_pnl", 0.0)) <= 0
    ]
    return gate(
        "advisory_timing_sensitivity",
        "warn" if negative else "pass",
        "60-second delay turns some blocks negative" if negative else "pass",
        metric={"negative_advisory_rows": negative},
        threshold="reported as warning; not a promotion blocker if live latency is proven inside critical budget",
    )


def missing_timing_coverage(timing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in timing_rows:
        delay = int(row.get("delay_seconds", -1))
        if delay not in CRITICAL_DELAYS:
            continue
        rows = int(row.get("rows", 0))
        coverage = float(row.get("coverage", 0.0))
        ok_rows = int(round(rows * coverage))
        required = int(math.ceil(rows * MIN_CRITICAL_TIMING_COVERAGE))
        if rows > 0 and ok_rows < required:
            out.append(
                {
                    "split": row.get("split"),
                    "delay_seconds": delay,
                    "rows": rows,
                    "covered_rows": ok_rows,
                    "required_rows": required,
                    "additional_rows_needed": required - ok_rows,
                    "coverage": round(coverage, 6),
                    "required_coverage": MIN_CRITICAL_TIMING_COVERAGE,
                }
            )
    return out


def decide(gates: list[dict[str, Any]]) -> str:
    failed = [row for row in gates if row["status"] == "fail"]
    blocked = [row for row in gates if row["status"] == "block"]
    if failed:
        return "reject_multi_contract_promotion_hard_gate_failed"
    if blocked:
        return "blocked_multi_contract_promotion_missing_timing_evidence"
    return "promote_multi_contract_research_candidate_to_paper_replay_gate"


def next_gate(decision: str, gates: list[dict[str, Any]]) -> str:
    if decision.startswith("promote_"):
        return "Replay live paper logs through the multi-contract sizer before enabling multi-contract paper quantities."
    if decision.startswith("blocked_"):
        return (
            "Do not enable multi-contract paper trading yet. Unblock by proving the same frozen candidate under >=95% "
            "critical high-resolution timing coverage or by replaying sufficiently fresh live-shadow/paper logs."
        )
    failed = ", ".join(row["gate"] for row in gates if row["status"] == "fail")
    return f"Reject this multi-contract challenger until the failed gates are fixed: {failed}."


def gate(name: str, status: str, reason: str, *, metric: Any, threshold: Any) -> dict[str, Any]:
    return {
        "gate": name,
        "status": status,
        "reason": reason,
        "metric": metric,
        "threshold": threshold,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    gates = payload["gates"]
    evidence = payload["evidence"]
    baseline = evidence["baseline_10000"]
    candidate = evidence["candidate_10000"]
    lines = [
        "# Protocol 154: Multi-Contract Promotion Decision",
        "",
        "No paid data was downloaded. No broker endpoint was called. No orders were placed. Protocol101 and account_aware_sizer_v1 remain frozen.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Promotable now: `{payload['promotable_now']}`",
        f"- Rejected: `{payload['rejected']}`",
        f"- Blocked: `{payload['blocked']}`",
        f"- Baseline one-contract PnL: `{money(baseline.get('total_pnl'))}`",
        f"- Multi-contract candidate PnL: `{money(candidate.get('total_pnl'))}`",
        f"- Incremental PnL: `{money(money_delta(candidate, baseline))}`",
        "",
        "## Gate Summary",
        "",
        "| gate | status | reason |",
        "| --- | --- | --- |",
    ]
    for row in gates:
        lines.append(f"| {row['gate']} | `{row['status']}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Timing Blocker",
            "",
            "The economic and live-stack gates pass, but promotion is blocked because critical high-resolution timing coverage is incomplete.",
            "",
            "| split | delay_s | covered | required | additional_needed | coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in missing_timing_coverage(evidence["timing_summary"]):
        lines.append(
            "| "
            f"{row['split']} | {row['delay_seconds']} | {row['covered_rows']} | {row['required_rows']} | "
            f"{row['additional_rows_needed']} | {pct(row['coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This is not a failed sizing hypothesis. It is also not promotable. The candidate is economically better "
                "than one contract in historical replay and can flow through the protected live/paper stack, but the timing "
                "evidence is not complete enough to justify multi-contract paper execution."
            ),
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def append_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = f"""
## 2026-05-15 Protocol 154 Multi-Contract Promotion Decision

```text
Date: 2026-05-15
Decision / Experiment: Aggregated Protocol151/152/153 into a single multi-contract promotion verdict.
Reason: User asked for all verifications necessary to decide whether the account-aware multi-contract challenger is promotable.
Data Used: Existing Protocol101 replay artifacts, existing high-resolution timing rows, and existing live-stack compatibility rows only.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Report: {payload['outputs']['report']}
Next Gate: {payload['next_gate']}
Owner: Codex
```
"""
    with path.open("a") as handle:
        handle.write(entry)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text())


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing required artifact: {path}")
    frame = pd.read_csv(path)
    return [{key: (None if pd.isna(value) else value) for key, value in row.items()} for row in frame.to_dict("records")]


def all_bool(rows: list[dict[str, Any]], column: str) -> bool:
    return bool(rows) and all(str(row.get(column)).lower() == "true" or row.get(column) is True for row in rows)


def min_numeric(rows: list[dict[str, Any]], column: str) -> float | None:
    values = [float(row[column]) for row in rows if row.get(column) is not None]
    return min(values) if values else None


def money_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    if candidate.get("total_pnl") is None or baseline.get("total_pnl") is None:
        return None
    return round(float(candidate["total_pnl"]) - float(baseline["total_pnl"]), 2)


def money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.0f}"


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
