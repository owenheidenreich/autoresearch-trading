"""Build candidate-level blocked-Protocol101 opportunity-cost labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_slot_opportunity_defer import ROLE_LABEL


DATASET_ROLE_LABEL = "DATASET_UNIFIED_SLOT_OPPORTUNITY_COST_LABELS_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_label_dataset")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_SLOT_OPPORTUNITY_COST_LABEL_DATASET.md")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_BASELINE_ACTIONS = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")
FLAT_COLUMNS = [
    "split",
    "session",
    "decision_time",
    "decision_dt",
    "candidate_uid",
    "contract_id",
    "right",
    "offset",
    "entry_premium",
    "candidate_exit_time",
    "candidate_exit_dt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
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
    sessions = included_session_keys(pd.read_csv(args.session_manifest))
    flat = load_flat(args.flat_dataset, sessions)
    baseline = load_baseline(args.baseline_actions, sessions, seed=int(args.seed))
    labels = build_labels(flat, baseline)
    labels_path = args.out_dir / "slot_opportunity_cost_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    split_summary = summarize(labels)
    split_summary.to_csv(args.out_dir / "split_summary.csv", index=False)
    payload = {
        "role_label": DATASET_ROLE_LABEL,
        "overlay_role_label": ROLE_LABEL,
        "what_is_this": "dataset / candidate-level blocked-Protocol101 opportunity-cost labels",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(labels),
        "seed": int(args.seed),
        "label_semantics": {
            "blocked_protocol101_entries": "Protocol101 entry events in the same session with current_decision_dt <= baseline_entry_dt < candidate_exit_dt.",
            "blocked_protocol101_pnl_*": "Sum of those baseline trade PnLs under deterministic slippage stress. This is a label, not a model input.",
            "forbidden_as_features": [
                "blocked_protocol101_entries",
                "blocked_protocol101_pnl_0_00",
                "blocked_protocol101_pnl_0_10",
                "blocked_protocol101_pnl_0_25",
                "candidate_exit_dt",
            ],
        },
        "row_counts": {
            "candidate_rows": int(len(labels)),
            "sessions": int(labels["session"].nunique()) if not labels.empty else 0,
            "positive_blocked_cost_rows": int((labels["blocked_protocol101_pnl_0_00"] > 0.0).sum()) if not labels.empty else 0,
        },
        "split_summary": split_summary.to_dict("records"),
        "next_required_evidence": [
            "Train/calibrate a causal opportunity-cost estimator using current-state features only.",
            "Use the estimator in strict replay as a defer overlay; do not use label columns as runtime inputs.",
            "Require nonnegative Q1/Q3 same-scope deltas under $0.00/$0.10/$0.25 stress before broader claims.",
        ],
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "labels": str(labels_path),
            "split_summary": str(args.out_dir / "split_summary.csv"),
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


def load_flat(path: Path, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=FLAT_COLUMNS)
    frame = filter_sessions(frame, sessions)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["candidate_exit_dt"] = pd.to_datetime(frame["candidate_exit_dt"], utc=True, errors="coerce")
    frame["entry_premium"] = pd.to_numeric(frame["entry_premium"], errors="coerce")
    return frame[frame["decision_dt"].notna() & frame["candidate_exit_dt"].notna()].sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)


def load_baseline(path: Path, sessions: set[tuple[str, str]], *, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int).eq(int(seed))].copy()
    frame = filter_sessions(frame, sessions)
    frame = frame[frame["protocol101_action"].astype(str).eq("enter")].copy()
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["baseline_trade_pnl"] = pd.to_numeric(frame["baseline_trade_pnl"], errors="coerce").fillna(0.0)
    return frame[frame["decision_dt"].notna()].sort_values(["split", "session", "decision_dt"]).reset_index(drop=True)


def build_labels(flat: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_by_session = {
        (str(split), str(session)): group.sort_values("decision_dt").copy()
        for (split, session), group in baseline.groupby(["split", "session"], sort=True)
    }
    rows = []
    for (split, session), group in flat.groupby(["split", "session"], sort=True):
        base = baseline_by_session.get((str(split), str(session)), pd.DataFrame())
        if base.empty:
            rows.append(group.assign(**empty_label_columns()))
            continue
        entry_times = base["decision_dt"].to_numpy(dtype="datetime64[ns]")
        pnl = base["baseline_trade_pnl"].to_numpy(dtype=float)
        current_action = base[["decision_dt", "protocol101_action", "contract_id", "surface_candidate_uid", "baseline_trade_pnl"]].rename(
            columns={
                "protocol101_action": "current_protocol101_action",
                "contract_id": "current_protocol101_contract_id",
                "surface_candidate_uid": "current_protocol101_surface_candidate_uid",
                "baseline_trade_pnl": "current_protocol101_trade_pnl",
            }
        )
        labeled = group.merge(current_action, on="decision_dt", how="left")
        labeled["current_protocol101_action"] = labeled["current_protocol101_action"].fillna("wait")
        labeled["current_protocol101_contract_id"] = labeled["current_protocol101_contract_id"].fillna("")
        labeled["current_protocol101_surface_candidate_uid"] = labeled["current_protocol101_surface_candidate_uid"].fillna("")
        labeled["current_protocol101_trade_pnl"] = pd.to_numeric(labeled["current_protocol101_trade_pnl"], errors="coerce").fillna(0.0)
        start_idx = np.searchsorted(entry_times, labeled["decision_dt"].to_numpy(dtype="datetime64[ns]"), side="left")
        end_idx = np.searchsorted(entry_times, labeled["candidate_exit_dt"].to_numpy(dtype="datetime64[ns]"), side="left")
        counts = np.maximum(0, end_idx - start_idx).astype(int)
        cumulative = np.concatenate([[0.0], np.cumsum(pnl, dtype=float)])
        pnl_0 = cumulative[end_idx] - cumulative[start_idx]
        pnl_10 = pnl_0 - counts * 20.0
        pnl_25 = pnl_0 - counts * 50.0
        labeled["blocked_protocol101_entries"] = counts
        labeled["blocked_protocol101_pnl_0_00"] = pnl_0
        labeled["blocked_protocol101_pnl_0_10"] = pnl_10
        labeled["blocked_protocol101_pnl_0_25"] = pnl_25
        labeled["has_blocked_protocol101_entry"] = labeled["blocked_protocol101_entries"].gt(0)
        rows.append(labeled)
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    keep = [
        "split",
        "session",
        "decision_time",
        "decision_dt",
        "candidate_uid",
        "contract_id",
        "right",
        "offset",
        "entry_premium",
        "candidate_exit_time",
        "candidate_exit_dt",
        "current_protocol101_action",
        "current_protocol101_contract_id",
        "current_protocol101_surface_candidate_uid",
        "current_protocol101_trade_pnl",
        "blocked_protocol101_entries",
        "blocked_protocol101_pnl_0_00",
        "blocked_protocol101_pnl_0_10",
        "blocked_protocol101_pnl_0_25",
        "has_blocked_protocol101_entry",
    ]
    return out[keep].sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)


def empty_label_columns() -> dict[str, Any]:
    return {
        "current_protocol101_action": "wait",
        "current_protocol101_contract_id": "",
        "current_protocol101_surface_candidate_uid": "",
        "current_protocol101_trade_pnl": 0.0,
        "blocked_protocol101_entries": 0,
        "blocked_protocol101_pnl_0_00": 0.0,
        "blocked_protocol101_pnl_0_10": 0.0,
        "blocked_protocol101_pnl_0_25": 0.0,
        "has_blocked_protocol101_entry": False,
    }


def summarize(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in labels.groupby("split", sort=True):
        rows.append(
            {
                "split": str(split),
                "rows": int(len(group)),
                "sessions": int(group["session"].nunique()),
                "blocked_entry_rows": int(group["has_blocked_protocol101_entry"].sum()),
                "blocked_entry_row_fraction": float(group["has_blocked_protocol101_entry"].mean()),
                "mean_blocked_protocol101_entries": float(group["blocked_protocol101_entries"].mean()),
                "mean_blocked_protocol101_pnl_0_00": float(group["blocked_protocol101_pnl_0_00"].mean()),
                "p95_blocked_protocol101_pnl_0_00": float(group["blocked_protocol101_pnl_0_00"].quantile(0.95)),
                "current_protocol101_enter_rows": int(group["current_protocol101_action"].astype(str).eq("enter").sum()),
            }
        )
    return pd.DataFrame(rows)


def decide(labels: pd.DataFrame) -> str:
    if labels.empty:
        return "slot_opportunity_cost_labels_blocked_no_rows"
    if int(labels["has_blocked_protocol101_entry"].sum()) <= 0:
        return "slot_opportunity_cost_labels_blocked_no_positive_cost_rows"
    return "slot_opportunity_cost_labels_ready_for_causal_estimator"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {DATASET_ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Label Semantics",
        "",
        f"- Blocked entries: {payload['label_semantics']['blocked_protocol101_entries']}",
        f"- Blocked PnL: {payload['label_semantics']['blocked_protocol101_pnl_*']}",
        "- These are labels, not runtime features.",
        "",
        "## Split Summary",
        "",
        table(payload["split_summary"], ["split", "rows", "sessions", "blocked_entry_rows", "blocked_entry_row_fraction", "mean_blocked_protocol101_pnl_0_00", "p95_blocked_protocol101_pnl_0_00"]),
        "",
        "## Next Required Evidence",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def included_session_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    included = frame[frame["included"].astype(bool)].copy()
    return set(zip(included["split"].astype(str), included["session"].astype(str)))


def filter_sessions(frame: pd.DataFrame, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    if frame.empty or not sessions:
        return frame.iloc[0:0].copy()
    allowed = pd.MultiIndex.from_tuples(sessions, names=["split", "session"])
    current = pd.MultiIndex.from_frame(frame[["split", "session"]].astype(str))
    return frame.loc[current.isin(allowed)].copy()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {DATASET_ROLE_LABEL}"
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
