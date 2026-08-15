"""Diagnose why the trained conservative flat-entry gate abstains."""
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

from v4.scripts.run_unified_conservative_neural_policy_strict_replay import (
    DEFAULT_FLAT_DATASET,
    DEFAULT_MODEL_ARTIFACTS,
    DEFAULT_SESSION_MANIFEST,
    attach_predictions,
    included_session_keys,
    load_flat_dataset,
    load_policy_bundle,
)


ROLE_LABEL = "DIAGNOSTIC_UNIFIED_CONSERVATIVE_FLAT_GATE_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifacts", type=Path, default=DEFAULT_MODEL_ARTIFACTS)
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_policy_bundle(args.model_artifacts)
    sessions = included_session_keys(pd.read_csv(args.session_manifest))
    flat = load_flat_dataset(args.flat_dataset, bundle["flat_feature_columns"], sessions)
    flat = attach_predictions(flat, bundle, head="flat")
    config = bundle["config"]
    flat["a_enter"] = pd.to_numeric(flat["a_enter"], errors="coerce") if "a_enter" in flat.columns else np.nan
    flat["target_positive"] = flat["a_enter"].gt(0.0)
    flat["pass_advantage_margin"] = flat["predicted_advantage"].ge(float(config.min_advantage_margin))
    flat["pass_positive_probability"] = flat["positive_probability"].ge(float(config.positive_probability_min))
    flat["pass_tail_probability"] = flat["tail_probability"].le(float(config.tail_probability_max))
    flat["pass_all_model_gates"] = flat["pass_advantage_margin"] & flat["pass_positive_probability"] & flat["pass_tail_probability"]

    split_rows = [summarize_gate_frame(split, group, config) for split, group in flat.groupby("split", sort=True)]
    total_row = summarize_gate_frame("total", flat, config)
    split_summary = pd.DataFrame([*split_rows, total_row])
    split_summary.to_csv(args.out_dir / "split_gate_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "diagnostic / flat-entry conservative gate abstention analysis",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(total_row),
        "diagnosis": diagnose(total_row),
        "policy_thresholds": {
            "min_advantage_margin": float(config.min_advantage_margin),
            "positive_probability_min": float(config.positive_probability_min),
            "tail_probability_max": float(config.tail_probability_max),
        },
        "split_gate_summary": split_summary.to_dict("records"),
        "next_required_evidence": next_required_evidence(total_row),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "split_gate_summary": str(args.out_dir / "split_gate_summary.csv"),
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


def summarize_gate_frame(split: str, frame: pd.DataFrame, config: Any) -> dict[str, Any]:
    if frame.empty:
        return {"split": split, "rows": 0}
    positive = frame["target_positive"].astype(bool)
    return {
        "split": str(split),
        "rows": int(len(frame)),
        "positive_label_rows": int(positive.sum()),
        "positive_label_rate": float(positive.mean()),
        "a_enter_mean": safe_mean(frame["a_enter"]),
        "a_enter_p95": quantile(frame["a_enter"], 0.95),
        "a_enter_p99": quantile(frame["a_enter"], 0.99),
        "predicted_advantage_mean": safe_mean(frame["predicted_advantage"]),
        "predicted_advantage_p95": quantile(frame["predicted_advantage"], 0.95),
        "predicted_advantage_p99": quantile(frame["predicted_advantage"], 0.99),
        "predicted_advantage_max": safe_max(frame["predicted_advantage"]),
        "positive_probability_mean": safe_mean(frame["positive_probability"]),
        "positive_probability_p95": quantile(frame["positive_probability"], 0.95),
        "positive_probability_p99": quantile(frame["positive_probability"], 0.99),
        "positive_probability_max": safe_max(frame["positive_probability"]),
        "tail_probability_mean": safe_mean(frame["tail_probability"]),
        "tail_probability_p95": quantile(frame["tail_probability"], 0.95),
        "tail_probability_p99": quantile(frame["tail_probability"], 0.99),
        "tail_probability_max": safe_max(frame["tail_probability"]),
        "pass_advantage_margin_rows": int(frame["pass_advantage_margin"].sum()),
        "pass_positive_probability_rows": int(frame["pass_positive_probability"].sum()),
        "pass_tail_probability_rows": int(frame["pass_tail_probability"].sum()),
        "pass_all_model_gates_rows": int(frame["pass_all_model_gates"].sum()),
        "positive_rows_pass_all_model_gates": int((positive & frame["pass_all_model_gates"]).sum()),
        "advantage_margin": float(config.min_advantage_margin),
        "positive_probability_min": float(config.positive_probability_min),
        "tail_probability_max": float(config.tail_probability_max),
    }


def decide(total: dict[str, Any]) -> str:
    if int(total.get("pass_all_model_gates_rows", 0)) == 0:
        return "flat_entry_gate_overconservative_zero_model_overrides"
    return "flat_entry_gate_produces_model_overrides_needs_replay_attribution"


def diagnose(total: dict[str, Any]) -> list[str]:
    rows = int(total.get("rows", 0))
    positive_rate = float(total.get("positive_label_rate", 0.0))
    pass_adv = int(total.get("pass_advantage_margin_rows", 0))
    pass_pos = int(total.get("pass_positive_probability_rows", 0))
    pass_tail = int(total.get("pass_tail_probability_rows", 0))
    pass_all = int(total.get("pass_all_model_gates_rows", 0))
    out = [
        f"Flat A_enter positives are rare: {positive_rate:.4%} of {rows:,} full-surface candidate rows.",
        f"Gate component pass counts: advantage={pass_adv:,}, positive_probability={pass_pos:,}, tail={pass_tail:,}, all={pass_all:,}.",
    ]
    if pass_all == 0:
        if pass_pos == 0:
            out.append("The positive-probability head is the binding bottleneck; class imbalance or calibration must be fixed before retraining/replay.")
        elif pass_adv == 0:
            out.append("The advantage regression head is the binding bottleneck; target scaling or loss balance must be fixed before retraining/replay.")
        else:
            out.append("No row satisfies all conservative gates at once; threshold calibration must be treated as a preregistered training-formulation issue.")
    return out


def next_required_evidence(total: dict[str, Any]) -> list[str]:
    if int(total.get("pass_all_model_gates_rows", 0)) == 0:
        return [
            "Before retraining, pre-register a class-imbalance/calibration fix for the flat-entry positive head.",
            "Report gate-component pass counts on train/validation/recent before running strict replay.",
            "Keep Protocol101 as the full fallback; do not loosen thresholds ad hoc to create trades.",
        ]
    return [
        "Run strict replay attribution for the generated overrides.",
        "Bucket overrides by side, time, moneyness, premium, and label advantage before any challenge claim.",
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
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["diagnosis"])
    lines.extend(
        [
            "",
            "## Gate Summary",
            "",
            "| split | rows | positive rate | adv pass | pos-prob pass | tail pass | all pass | pred adv p99 | pos prob max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["split_gate_summary"]:
        lines.append(
            f"| {row.get('split')} | {row.get('rows', 0)} | {row.get('positive_label_rate', 0.0):.4f} | "
            f"{row.get('pass_advantage_margin_rows', 0)} | {row.get('pass_positive_probability_rows', 0)} | "
            f"{row.get('pass_tail_probability_rows', 0)} | {row.get('pass_all_model_gates_rows', 0)} | "
            f"{row.get('predicted_advantage_p99', 0.0):.2f} | {row.get('positive_probability_max', 0.0):.4f} |"
        )
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


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


def quantile(series: pd.Series, q: float) -> float:
    return none_if_nan(pd.to_numeric(series, errors="coerce").quantile(q))


def safe_mean(series: pd.Series) -> float:
    return none_if_nan(pd.to_numeric(series, errors="coerce").mean())


def safe_max(series: pd.Series) -> float:
    return none_if_nan(pd.to_numeric(series, errors="coerce").max())


def none_if_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


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
