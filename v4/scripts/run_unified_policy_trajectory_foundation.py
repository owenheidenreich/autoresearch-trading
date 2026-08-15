"""Audit/build the unified policy trajectory dataset foundation.

This runner does not train a model and does not duplicate the full training
dataset. It verifies that existing flat and holding action-advantage artifacts
can be interpreted as `UnifiedPolicyTrajectoryDatasetV1` inputs, emits coverage
summaries, and records the remaining gates that still block training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_policy_trajectory import (
    FLAT_STATE_SOURCE,
    HOLDING_STATE_SOURCE,
    ROLE_LABEL,
    TRAJECTORY_DATASET_CONTRACT,
    flat_decision_states_from_action_rows,
    holding_decision_states_from_rows,
    summarize_flat_coverage,
    summarize_holding_coverage,
    trajectory_contract_payload,
    trajectory_foundation_decision,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_policy_trajectory_foundation")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_POLICY_TRAJECTORY_FOUNDATION.md")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_HOLDING_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_FOUNDATION_AUDIT = Path("v4/audit/autoresearch/foundation_hardening_review/summary.json")
DEFAULT_PROTOCOL276_ATTRIBUTION = Path("v4/audit/autoresearch/protocol276_integrated_lifecycle_failure_attribution/summary.json")

FLAT_COLUMNS = [
    "split",
    "session",
    "decision_time",
    "decision_dt",
    "candidate_uid",
    "trade_uid",
    "contract_id",
    "root",
    "settlement_style",
    "right",
    "offset",
    "entry_quote_time",
    "entry_bid",
    "entry_ask",
    "entry_mid",
    "entry_spread",
    "entry_bid_size",
    "entry_ask_size",
    "entry_premium",
    "entry_delta",
    "entry_gamma",
    "entry_theta",
    "entry_iv",
    "oracle_action",
    "q_wait",
    "q_enter",
    "a_enter",
    "future_path_columns_used_as_features",
]
HOLDING_COLUMNS = [
    "split",
    "session",
    "candidate_uid",
    "trade_uid",
    "contract_id",
    "right",
    "entry_time",
    "state_time",
    "state_index",
    "entry_ask",
    "bid",
    "ask",
    "current_pnl",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
    "minutes_since_entry",
    "minutes_to_forced_flat",
    "oracle_holding_action",
    "q_exit",
    "q_hold",
    "a_hold",
    "future_path_columns_used_as_features",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--holding-dataset", type=Path, default=DEFAULT_HOLDING_DATASET)
    parser.add_argument("--foundation-audit", type=Path, default=DEFAULT_FOUNDATION_AUDIT)
    parser.add_argument("--protocol276-attribution", type=Path, default=DEFAULT_PROTOCOL276_ATTRIBUTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--sample-flat-events", type=int, default=3)
    parser.add_argument("--sample-holding-rows", type=int, default=3)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flat = load_parquet_columns(args.flat_dataset, FLAT_COLUMNS)
    holding = load_parquet_columns(args.holding_dataset, HOLDING_COLUMNS)
    feature_contract = trajectory_contract_payload()
    foundation = load_json(args.foundation_audit)
    p276 = load_json(args.protocol276_attribution)
    gates = foundation_gates(foundation)
    flat_coverage = summarize_flat_coverage(flat)
    holding_coverage = summarize_holding_coverage(holding)
    sample_flat = flat_decision_states_from_action_rows(flat, max_events=int(args.sample_flat_events))
    sample_holding = holding_decision_states_from_rows(holding, max_rows=int(args.sample_holding_rows))
    decision = trajectory_foundation_decision(
        flat_rows=int(len(flat)),
        holding_rows=int(len(holding)),
        feature_contract_status=feature_contract["feature_contract"]["status"],
        fill_model_ready=gates["fill_model_ready"],
        untouched_holdout_ready=gates["untouched_holdout_ready"],
        live_parity_ready=gates["live_parity_ready"],
    )
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "dataset foundation / unified conservative policy trajectory manifest",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "trajectory_contract": TRAJECTORY_DATASET_CONTRACT,
        "flat_state_source": FLAT_STATE_SOURCE,
        "holding_state_source": HOLDING_STATE_SOURCE,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decision,
        "row_counts": {
            "flat_candidate_rows": int(len(flat)),
            "holding_state_rows": int(len(holding)),
            "sample_flat_states": int(len(sample_flat)),
            "sample_holding_states": int(len(sample_holding)),
        },
        "foundation_gates": gates,
        "feature_contract": feature_contract["feature_contract"],
        "protocol276_evidence": p276.get("root_cause_summary", []),
        "sample_states": {
            "flat": [asdict(state) for state in sample_flat],
            "holding": [asdict(state) for state in sample_holding],
        },
        "next_allowed_work": [
            "Materialize a versioned trajectory dataset only after choosing a storage layout and untouched block.",
            "Expand the DP oracle so flat wait/enter and holding hold/exit values are computed on the same serial account trajectory.",
            "Attach Protocol101 baseline actions to every trajectory event for conservative policy improvement.",
            "Keep neural training blocked until the readiness packet clears the DP oracle, Protocol101 baseline, fill, and live no-order parity gates.",
        ],
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "contract": str(args.out_dir / "trajectory_dataset_contract.json"),
            "flat_coverage": str(args.out_dir / "flat_coverage_by_split.csv"),
            "holding_coverage": str(args.out_dir / "holding_coverage_by_split.csv"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
    }
    write_json(args.out_dir / "summary.json", payload)
    write_json(args.out_dir / "trajectory_dataset_contract.json", feature_contract)
    flat_coverage.to_csv(args.out_dir / "flat_coverage_by_split.csv", index=False)
    holding_coverage.to_csv(args.out_dir / "holding_coverage_by_split.csv", index=False)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns)


def foundation_gates(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {str(item.get("name")): str(item.get("status")) for item in summary.get("foundation_checks", [])}
    return {
        "fill_model_ready": checks.get("Fill model calibration") == "pass",
        "untouched_holdout_ready": checks.get("Untouched holdout reservation") == "pass",
        "live_parity_ready": checks.get("Challenger runtime parity") == "pass",
        "source_decision": summary.get("decision", "missing"),
        "raw_check_statuses": checks,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Summary",
        "",
        f"The existing flat and holding artifacts can be read as `{TRAJECTORY_DATASET_CONTRACT}` sources, but this remains a foundation manifest. Neural training is still controlled by the unified readiness packet.",
        "",
        "## Coverage",
        "",
        f"- Flat candidate rows: `{payload['row_counts']['flat_candidate_rows']}` from `{payload['flat_state_source']}`",
        f"- Holding state rows: `{payload['row_counts']['holding_state_rows']}` from `{payload['holding_state_source']}`",
        f"- Sample flat states built: `{payload['row_counts']['sample_flat_states']}`",
        f"- Sample holding states built: `{payload['row_counts']['sample_holding_states']}`",
        "",
        "## Foundation Gates",
        "",
        f"- Fill model ready: `{payload['foundation_gates']['fill_model_ready']}`",
        f"- Untouched holdout ready: `{payload['foundation_gates']['untouched_holdout_ready']}`",
        f"- Live no-order parity ready: `{payload['foundation_gates']['live_parity_ready']}`",
        "",
        "## Protocol276 Evidence Kept",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["protocol276_evidence"])
    lines.extend(["", "## Next Allowed Work", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_allowed_work"], start=1))
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{payload['outputs']['summary']}`",
            f"- Report: `{payload['outputs']['report']}`",
            f"- Contract: `{payload['outputs']['contract']}`",
            f"- Flat coverage: `{payload['outputs']['flat_coverage']}`",
            f"- Holding coverage: `{payload['outputs']['holding_coverage']}`",
            f"- Docs copy: `{payload['outputs']['doc']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
    text = ledger.read_text()
    if marker in text:
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
