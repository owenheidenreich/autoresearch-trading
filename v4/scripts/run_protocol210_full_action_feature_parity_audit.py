"""AUDIT_FULL_ACTION_FEATURE_PARITY_V1.

Historically Protocol210. This audit checks whether the full-action challenger
dataset has the live-safe feature inputs that made PAPER_DEFAULT_PROTOCOL101
strong: surface edge plus causal short-history context.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.model.serial_opportunity import ENTRY_FEATURE_COLUMNS
from v4.scripts.run_protocol101_event_history_policy import HISTORY_FEATURE_COLUMNS
from v4.scripts.run_protocol185_surface_edge_full_action_enrichment import SURFACE_FEATURE_COLUMNS


ROLE_LABEL = "AUDIT_FULL_ACTION_FEATURE_PARITY_V1"
HISTORICAL_ID = "Protocol210"
DEFAULT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_189_full_coverage_surface_edge_enrichment/protocol185_full_action_with_surface_edge.parquet")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_210_full_action_feature_parity_audit")

EQUIVALENT_FEATURES = {
    "edge": "surface_edge",
    "hist_prev_max_edge": "surface_prev_best_edge",
    "hist_prev_mean_edge": "surface_prev_mean_edge",
    "hist_roll3_max_edge": "surface_roll3_best_edge",
    "hist_roll3_mean_edge": "surface_roll3_mean_edge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset)
    columns = set(dataset.columns)
    required = list(dict.fromkeys([*ENTRY_FEATURE_COLUMNS, *HISTORY_FEATURE_COLUMNS]))
    rows = []
    missing = []
    equivalent = []
    present = []
    for feature in required:
        if feature in columns:
            status = "present"
            present.append(feature)
            mapped = feature
        elif feature in EQUIVALENT_FEATURES and EQUIVALENT_FEATURES[feature] in columns:
            status = "equivalent_present"
            equivalent.append(feature)
            mapped = EQUIVALENT_FEATURES[feature]
        else:
            status = "missing"
            missing.append(feature)
            mapped = ""
        rows.append({"protocol101_feature": feature, "status": status, "full_action_column": mapped})
    surface_missing = [feature for feature in SURFACE_FEATURE_COLUMNS if feature not in columns]
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_FULL_ACTION_SURFACE_EDGE_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "other_baseline_label": "none",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "present_protocol101_features": present,
        "equivalent_protocol101_features": equivalent,
        "missing_protocol101_features": missing,
        "missing_surface_features": surface_missing,
        "feature_rows": rows,
        "decision": "repair_missing_causal_history_features" if missing else "full_action_feature_parity_pass",
        "next_experiment": (
            "Add the missing causal short-history features to the full-action surface-edge dataset, then rerun "
            "the two-stage full-action policy with the repaired feature set."
            if missing
            else "Proceed to no-order runtime parity for the full-action surface-edge challenger."
        ),
    }
    pd.DataFrame(rows).to_csv(args.out_dir / "feature_parity_rows.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "missing": len(missing), "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['role_label']}",
        "",
        f"What is this: {payload['what_is_this']}",
        f"Does it change the paper-trading default: {'yes' if payload['changes_paper_default'] else 'no'}",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Other baseline: {payload['other_baseline_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Summary",
        "",
        f"- Rows audited: `{payload['rows']}`",
        f"- Present Protocol101 features: `{len(payload['present_protocol101_features'])}`",
        f"- Equivalent Protocol101 features: `{len(payload['equivalent_protocol101_features'])}`",
        f"- Missing Protocol101 features: `{len(payload['missing_protocol101_features'])}`",
        f"- Missing surface features: `{len(payload['missing_surface_features'])}`",
        "",
        "## Missing Protocol101-Style Features",
        "",
    ]
    if payload["missing_protocol101_features"]:
        lines.extend(f"- `{feature}`" for feature in payload["missing_protocol101_features"])
    else:
        lines.append("_None._")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Feature rows: `{path.parent / 'feature_parity_rows.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

