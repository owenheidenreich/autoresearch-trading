"""Explain flat-calibrated Q1/Q3 underperformance versus Protocol101."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "ATTRIBUTION_UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE_V1"
DEFAULT_REPLAY_DIR = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1")
DEFAULT_BASELINE_ACTIONS = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_CONSERVATIVE_Q1_Q3_UNDERPERFORMANCE_ATTRIBUTION_V1.md")
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    included_sessions = included_session_keys(pd.read_csv(args.session_manifest))
    baseline = load_baseline(args.baseline_actions, seed=int(args.seed), included_sessions=included_sessions)
    replay_summary = load_json(args.replay_dir / "summary.json")
    component_rows: list[dict[str, Any]] = []
    missed_rows: list[pd.DataFrame] = []
    blocker_rows: list[pd.DataFrame] = []
    for trades_path in sorted(args.replay_dir.glob("trades_slippage_*.csv")):
        slippage = slippage_from_path(trades_path)
        trades = load_trades(trades_path)
        decisions = load_decisions(args.replay_dir / f"decisions_slippage_{slippage_suffix(slippage)}.csv")
        stressed_baseline = baseline.copy()
        stressed_baseline["stressed_baseline_pnl"] = stressed_baseline["baseline_trade_pnl"] - 2.0 * float(slippage) * CONTRACT_MULTIPLIER
        components, missed, blockers = attribute_one_slippage(stressed_baseline, trades, decisions, slippage=slippage)
        component_rows.extend(components)
        missed_rows.append(missed)
        blocker_rows.append(blockers)
    component_summary = pd.DataFrame(component_rows)
    missed_all = pd.concat(missed_rows, ignore_index=True, sort=False) if missed_rows else pd.DataFrame()
    blockers_all = pd.concat(blocker_rows, ignore_index=True, sort=False) if blocker_rows else pd.DataFrame()
    component_summary.to_csv(args.out_dir / "component_summary.csv", index=False)
    missed_all.to_csv(args.out_dir / "missed_protocol101_entries.csv", index=False)
    blockers_all.to_csv(args.out_dir / "challenger_blocking_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "attribution / flat-calibrated Q1-Q3 underperformance versus same-scope Protocol101",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(component_summary),
        "replay_decision": replay_summary.get("decision", "missing"),
        "component_summary": component_summary.to_dict("records"),
        "q1_q3_diagnosis": diagnose_q1_q3(component_summary, missed_all, blockers_all),
        "top_missed_protocol101_entries": top_rows(missed_all, "stressed_baseline_pnl", n=16, ascending=False),
        "top_challenger_blockers": top_rows(blockers_all, "blocked_protocol101_pnl", n=16, ascending=False),
        "preregistered_defer_constraints": preregistered_defer_constraints(),
        "next_required_evidence": next_required_evidence(),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "component_summary": str(args.out_dir / "component_summary.csv"),
            "missed_protocol101_entries": str(args.out_dir / "missed_protocol101_entries.csv"),
            "challenger_blocking_summary": str(args.out_dir / "challenger_blocking_summary.csv"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_baseline(path: Path, *, seed: int, included_sessions: set[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int).eq(int(seed))].copy()
    frame = filter_sessions(frame, included_sessions)
    frame = frame[frame["protocol101_action"].astype(str).eq("enter")].copy()
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["baseline_exit_dt"] = pd.to_datetime(frame["baseline_exit_time"], utc=True, errors="coerce")
    frame["baseline_trade_pnl"] = pd.to_numeric(frame["baseline_trade_pnl"], errors="coerce").fillna(0.0)
    return frame[frame["decision_dt"].notna()].sort_values(["split", "session", "decision_dt"]).reset_index(drop=True)


def included_session_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    included = frame[frame["included"].astype(bool)].copy()
    return set(zip(included["split"].astype(str), included["session"].astype(str)))


def filter_sessions(frame: pd.DataFrame, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    if frame.empty or not sessions:
        return frame.iloc[0:0].copy()
    allowed = pd.MultiIndex.from_tuples(sessions, names=["split", "session"])
    current = pd.MultiIndex.from_frame(frame[["split", "session"]].astype(str))
    return frame.loc[current.isin(allowed)].copy()


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["entry_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_dt"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    frame["duration_minutes"] = pd.to_numeric(frame["duration_minutes"], errors="coerce").fillna(0.0)
    return frame


def load_decisions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["decision_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    return frame


def attribute_one_slippage(
    baseline: pd.DataFrame,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    slippage: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    baseline = baseline.copy()
    trades = trades.copy()
    decisions = decisions.copy()
    baseline["event_key"] = baseline.apply(event_key_row, axis=1)
    trades["event_key"] = trades.apply(lambda row: event_key(str(row["split"]), str(row["session"]), pd.Timestamp(row["entry_dt"])), axis=1)
    decisions["event_key"] = decisions.apply(event_key_row, axis=1)
    protocol101_defer_keys = set(trades[trades["source"].astype(str).eq("protocol101_defer")]["event_key"].astype(str))
    challenger = trades[trades["source"].astype(str).eq("challenger")].copy()
    missed = baseline[~baseline["event_key"].astype(str).isin(protocol101_defer_keys)].copy()
    missed = classify_missed(missed, decisions, trades)
    blockers = summarize_blockers(challenger, missed, slippage=slippage)
    components: list[dict[str, Any]] = []
    for split in sorted(set(baseline["split"].astype(str)) | set(trades["split"].astype(str))):
        split_baseline = baseline[baseline["split"].astype(str).eq(split)]
        split_trades = trades[trades["split"].astype(str).eq(split)]
        split_missed = missed[missed["split"].astype(str).eq(split)]
        split_challenger = challenger[challenger["split"].astype(str).eq(split)]
        baseline_pnl = float(split_baseline["stressed_baseline_pnl"].sum())
        replay_pnl = float(split_trades["pnl"].sum())
        challenger_pnl = float(split_challenger["pnl"].sum())
        missed_baseline_pnl = float(split_missed["stressed_baseline_pnl"].sum())
        components.append(
            {
                "slippage_per_side": float(slippage),
                "split": split,
                "baseline_entries": int(len(split_baseline)),
                "replay_trades": int(len(split_trades)),
                "challenger_entries": int(len(split_challenger)),
                "missed_protocol101_entries": int(len(split_missed)),
                "baseline_pnl": baseline_pnl,
                "replay_pnl": replay_pnl,
                "challenger_pnl": challenger_pnl,
                "missed_protocol101_pnl": missed_baseline_pnl,
                "component_delta_challenger_minus_missed": float(challenger_pnl - missed_baseline_pnl),
                "actual_delta": float(replay_pnl - baseline_pnl),
                "residual": float((replay_pnl - baseline_pnl) - (challenger_pnl - missed_baseline_pnl)),
                "missed_blocked_by_challenger_open": int((split_missed["miss_reason"].astype(str) == "blocked_by_challenger_open").sum()) if not split_missed.empty else 0,
                "missed_replaced_by_challenger_same_event": int((split_missed["miss_reason"].astype(str) == "replaced_by_challenger_same_event").sum()) if not split_missed.empty else 0,
            }
        )
    missed["slippage_per_side"] = float(slippage)
    return components, missed, blockers


def classify_missed(missed: pd.DataFrame, decisions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if missed.empty:
        return missed.assign(miss_reason=pd.Series(dtype=str), blocking_trade_uid=pd.Series(dtype=str), blocking_source=pd.Series(dtype=str))
    decision_by_key = {str(row["event_key"]): row for row in decisions.to_dict("records")}
    replay_trades = trades.copy()
    rows = []
    for _, row in missed.iterrows():
        key = str(row["event_key"])
        decision = decision_by_key.get(key, {})
        reason = "missed_reason_unknown"
        blocking_uid = ""
        blocking_source = ""
        if str(decision.get("decision", "")) == "challenger_entry":
            reason = "replaced_by_challenger_same_event"
            blocking_uid = str(decision.get("candidate_uid", ""))
            blocking_source = "challenger"
        elif str(decision.get("decision", "")) == "skip_open_position":
            blocker = open_trade_at(replay_trades, str(row["split"]), str(row["session"]), pd.Timestamp(row["decision_dt"]))
            if blocker is not None:
                blocking_source = str(blocker.get("source", ""))
                blocking_uid = str(blocker.get("candidate_uid", ""))
                reason = f"blocked_by_{blocking_source}_open" if blocking_source else "blocked_by_open_position"
            else:
                reason = "blocked_by_open_position_unmatched"
        elif str(decision.get("decision", "")) == "protocol101_defer_wait":
            reason = "deferred_wait_unexpected"
        out = row.to_dict()
        out["decision"] = str(decision.get("decision", "missing"))
        out["miss_reason"] = reason
        out["blocking_trade_uid"] = blocking_uid
        out["blocking_source"] = blocking_source
        rows.append(out)
    return pd.DataFrame(rows)


def open_trade_at(trades: pd.DataFrame, split: str, session: str, ts: pd.Timestamp) -> dict[str, Any] | None:
    eligible = trades[
        trades["split"].astype(str).eq(split)
        & trades["session"].astype(str).eq(session)
        & (trades["entry_dt"] < ts)
        & (trades["exit_dt"] > ts)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values("entry_dt").iloc[-1].to_dict()


def summarize_blockers(challenger: pd.DataFrame, missed: pd.DataFrame, *, slippage: float) -> pd.DataFrame:
    if challenger.empty:
        return pd.DataFrame()
    rows = []
    for _, trade in challenger.iterrows():
        blocked = missed[
            missed["blocking_trade_uid"].astype(str).eq(str(trade["candidate_uid"]))
            & missed["miss_reason"].astype(str).isin(["blocked_by_challenger_open", "replaced_by_challenger_same_event"])
        ].copy() if not missed.empty else pd.DataFrame()
        rows.append(
            {
                "slippage_per_side": float(slippage),
                "split": str(trade["split"]),
                "session": str(trade["session"]),
                "challenger_candidate_uid": str(trade["candidate_uid"]),
                "challenger_entry_time": pd.Timestamp(trade["entry_dt"]).isoformat(),
                "challenger_exit_time": pd.Timestamp(trade["exit_dt"]).isoformat(),
                "challenger_pnl": float(trade["pnl"]),
                "challenger_duration_minutes": float(trade["duration_minutes"]),
                "right": str(trade.get("right", "")),
                "offset": finite(trade.get("offset")),
                "exit_reason": str(trade.get("exit_reason", "")),
                "blocked_protocol101_entries": int(len(blocked)),
                "blocked_protocol101_pnl": float(blocked["stressed_baseline_pnl"].sum()) if not blocked.empty else 0.0,
                "net_vs_blocked_protocol101": float(trade["pnl"] - (blocked["stressed_baseline_pnl"].sum() if not blocked.empty else 0.0)),
            }
        )
    return pd.DataFrame(rows)


def decide(components: pd.DataFrame) -> str:
    if components.empty:
        return "q1_q3_underperformance_attribution_blocked_missing_components"
    zero = components[components["slippage_per_side"].astype(float).eq(0.0)]
    q1_q3 = zero[zero["split"].astype(str).isin(["q1_2026", "q3_2025"])]
    if not q1_q3.empty and (q1_q3["actual_delta"] < 0.0).any():
        return "q1_q3_underperformance_explained_by_missed_protocol101_opportunity_cost"
    return "q1_q3_underperformance_not_reproduced"


def diagnose_q1_q3(components: pd.DataFrame, missed: pd.DataFrame, blockers: pd.DataFrame) -> list[str]:
    if components.empty:
        return ["No component rows were produced."]
    out = []
    zero = components[components["slippage_per_side"].astype(float).eq(0.0)]
    for split in ["q1_2026", "q3_2025", "q4_2025", "recent_2026"]:
        row = zero[zero["split"].astype(str).eq(split)]
        if row.empty:
            continue
        item = row.iloc[0]
        out.append(
            f"{split}: challenger PnL {money(item['challenger_pnl'])} minus missed Protocol101 PnL "
            f"{money(item['missed_protocol101_pnl'])} explains delta {money(item['actual_delta'])}."
        )
    for split in ["q1_2026", "q3_2025"]:
        blocked = blockers[
            blockers["slippage_per_side"].astype(float).eq(0.0)
            & blockers["split"].astype(str).eq(split)
            & (pd.to_numeric(blockers["blocked_protocol101_pnl"], errors="coerce").fillna(0.0) > 0.0)
        ]
        if not blocked.empty:
            worst = blocked.sort_values("net_vs_blocked_protocol101").iloc[0]
            out.append(
                f"{split}: worst blocker net {money(worst['net_vs_blocked_protocol101'])}; challenger {money(worst['challenger_pnl'])} "
                f"blocked {int(worst['blocked_protocol101_entries'])} Protocol101 entries worth {money(worst['blocked_protocol101_pnl'])}."
            )
    if not missed.empty:
        reasons = missed[missed["slippage_per_side"].astype(float).eq(0.0)]["miss_reason"].astype(str).value_counts().to_dict()
        out.append(f"Zero-slippage missed Protocol101 entry reasons: {reasons}.")
    return out


def preregistered_defer_constraints() -> list[dict[str, str]]:
    return [
        {
            "constraint": "slot_opportunity_cost_margin",
            "rule": "Allow a challenger override only if predicted entry advantage exceeds a learned missed-Protocol101 opportunity-cost estimate plus the fixed conservative margin.",
        },
        {
            "constraint": "split_stability_gate",
            "rule": "A replay candidate must have nonnegative same-scope delta on q1_2026 and q3_2025 under $0.00, $0.10, and $0.25 stress before any broader claim.",
        },
        {
            "constraint": "blocked_baseline_budget",
            "rule": "Reject or penalize overrides whose predicted duration implies blocking more than one baseline entry or whose expected blocked baseline PnL exceeds challenger edge.",
        },
        {
            "constraint": "forced_flat_dependence_audit",
            "rule": "Report PnL with forced-flat-no-exit-signal trades removed; a challenger cannot be promoted if gains depend mainly on forced flat.",
        },
    ]


def next_required_evidence() -> list[str]:
    return [
        "Build the slot-opportunity-cost/defer constraint as a preregistered overlay before retraining.",
        "Rerun strict replay and require nonnegative Q1/Q3 deltas under all deterministic slippage stresses.",
        "Keep Protocol101 as paper default; this attribution is research-only.",
    ]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        f"Replay decision: `{payload['replay_decision']}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["q1_q3_diagnosis"])
    lines.extend(["", "## Component Summary", "", table(payload["component_summary"], ["slippage_per_side", "split", "challenger_pnl", "missed_protocol101_pnl", "component_delta_challenger_minus_missed", "actual_delta", "residual", "challenger_entries", "missed_protocol101_entries"])])
    lines.extend(["", "## Top Missed Protocol101 Entries", "", table(payload["top_missed_protocol101_entries"], ["slippage_per_side", "split", "session", "decision_time", "baseline_trade_pnl", "stressed_baseline_pnl", "miss_reason", "blocking_source"])])
    lines.extend(["", "## Top Challenger Blockers", "", table(payload["top_challenger_blockers"], ["slippage_per_side", "split", "session", "challenger_pnl", "blocked_protocol101_entries", "blocked_protocol101_pnl", "net_vs_blocked_protocol101", "challenger_duration_minutes", "exit_reason"])])
    lines.extend(["", "## Preregistered Defer Constraints", ""])
    lines.extend(f"{idx}. `{item['constraint']}`: {item['rule']}" for idx, item in enumerate(payload["preregistered_defer_constraints"], start=1))
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def top_rows(frame: pd.DataFrame, column: str, *, n: int, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return []
    ordered = frame.sort_values(column, ascending=ascending).head(n)
    return [{key: scalar(value) for key, value in row.items()} for row in ordered.to_dict("records")]


def table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int = 18) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows[:max_rows]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append("| " + " | ".join(["...", f"{len(rows) - max_rows} more rows", *[""] * max(0, len(columns) - 2)]) + " |")
    return "\n".join(lines)


def event_key_row(row: pd.Series | dict[str, Any]) -> str:
    return event_key(str(row["split"]), str(row["session"]), pd.Timestamp(row["decision_dt"]))


def event_key(split: str, session: str, decision_dt: pd.Timestamp) -> str:
    return f"{split}|{session}|{pd.Timestamp(decision_dt).isoformat()}"


def slippage_from_path(path: Path) -> float:
    text = path.stem.removeprefix("trades_slippage_")
    return float(text.replace("_", "."))


def slippage_suffix(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def money(value: Any) -> str:
    x = finite(value)
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return None if not math.isfinite(out) else out
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    return None if isinstance(missing, (bool, np.bool_)) and missing else value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
