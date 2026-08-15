"""Build the slot opportunity-cost defer overlay foundation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_slot_opportunity_defer import (
    ROLE_LABEL,
    SlotOpportunityDeferConfig,
    oracle_net_vs_blocked_protocol101,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY_FOUNDATION.md")
DEFAULT_UNDERPERFORMANCE = Path("v4/audit/autoresearch/unified_conservative_q1_q3_underperformance_attribution_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--underperformance-dir", type=Path, default=DEFAULT_UNDERPERFORMANCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--min-realized-net", type=float, default=0.0)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    underperformance = load_json(args.underperformance_dir / "summary.json")
    blockers = pd.read_csv(args.underperformance_dir / "challenger_blocking_summary.csv")
    components = pd.read_csv(args.underperformance_dir / "component_summary.csv")
    blockers = enrich_blockers(blockers)
    oracle = oracle_overlay_summary(blockers, min_realized_net=float(args.min_realized_net))
    oracle.to_csv(args.out_dir / "oracle_overlay_summary.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "foundation / slot opportunity-cost defer overlay contract and oracle diagnostic",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decide(oracle),
        "underperformance_decision": underperformance.get("decision", "missing"),
        "config": SlotOpportunityDeferConfig().to_dict(),
        "min_realized_net_for_oracle_keep": float(args.min_realized_net),
        "important_guardrail": "The oracle overlay uses realized blocked Protocol101 PnL and is diagnostic only; it is not live-deployable.",
        "causal_overlay_contract": causal_overlay_contract(),
        "oracle_overlay_summary": oracle.to_dict("records"),
        "original_component_summary": components.to_dict("records"),
        "diagnosis": diagnose(oracle),
        "next_required_evidence": next_required_evidence(),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "oracle_overlay_summary": str(args.out_dir / "oracle_overlay_summary.csv"),
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


def enrich_blockers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["challenger_pnl", "blocked_protocol101_pnl", "blocked_protocol101_entries", "slippage_per_side"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["oracle_net_vs_blocked_protocol101"] = [
        oracle_net_vs_blocked_protocol101(challenger_pnl=row.challenger_pnl, blocked_protocol101_pnl=row.blocked_protocol101_pnl)
        for row in out.itertuples(index=False)
    ]
    return out


def oracle_overlay_summary(blockers: pd.DataFrame, *, min_realized_net: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (slippage, split), group in blockers.groupby(["slippage_per_side", "split"], sort=True):
        kept = group[group["oracle_net_vs_blocked_protocol101"].ge(float(min_realized_net))]
        rejected = group[~group.index.isin(kept.index)]
        rows.append(
            {
                "slippage_per_side": float(slippage),
                "split": str(split),
                "challenger_entries": int(len(group)),
                "oracle_kept_entries": int(len(kept)),
                "oracle_rejected_entries": int(len(rejected)),
                "original_challenger_pnl": float(group["challenger_pnl"].sum()),
                "original_blocked_protocol101_pnl": float(group["blocked_protocol101_pnl"].sum()),
                "original_delta_vs_protocol101": float(group["oracle_net_vs_blocked_protocol101"].sum()),
                "oracle_overlay_delta_vs_protocol101": float(kept["oracle_net_vs_blocked_protocol101"].sum()),
                "negative_net_rejected": float(rejected["oracle_net_vs_blocked_protocol101"].clip(upper=0.0).sum()),
                "blocked_protocol101_pnl_released": float(rejected["blocked_protocol101_pnl"].sum()),
            }
        )
    return pd.DataFrame(rows)


def decide(oracle: pd.DataFrame) -> str:
    if oracle.empty:
        return "slot_opportunity_defer_overlay_foundation_blocked_missing_oracle_rows"
    q1_q3 = oracle[oracle["split"].astype(str).isin(["q1_2026", "q3_2025"])]
    if not q1_q3.empty and (q1_q3["oracle_overlay_delta_vs_protocol101"] >= 0.0).all():
        return "slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator"
    return "slot_opportunity_defer_overlay_oracle_target_still_fails_q1_q3"


def causal_overlay_contract() -> dict[str, Any]:
    return {
        "decision_rule": "allow challenger only if predicted challenger advantage - estimated blocked Protocol101 opportunity cost - uncertainty >= fixed margin",
        "allowed_causal_inputs": [
            "current Protocol101 baseline action and candidate",
            "recent Protocol101 entry cadence features",
            "recent full-surface opportunity density",
            "candidate predicted lifecycle duration / exit confidence",
            "account state, time-to-close, premium, side, moneyness, Greeks, quote freshness",
        ],
        "forbidden_inputs": [
            "future Protocol101 entries",
            "realized blocked Protocol101 PnL",
            "realized challenger exit or duration",
            "future bid/ask path",
        ],
        "promotion_gate": "nonnegative q1_2026 and q3_2025 same-scope deltas under $0.00/$0.10/$0.25 stress before broader claims",
    }


def diagnose(oracle: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in oracle.iterrows():
        rows.append(
            f"{row['split']} @ {row['slippage_per_side']:.2f}: oracle overlay keeps {int(row['oracle_kept_entries'])}/"
            f"{int(row['challenger_entries'])} overrides and changes delta from {money(row['original_delta_vs_protocol101'])} "
            f"to {money(row['oracle_overlay_delta_vs_protocol101'])}."
        )
    return rows


def next_required_evidence() -> list[str]:
    return [
        "Materialize causal blocked-Protocol101 opportunity-cost labels for candidate override events.",
        "Train or calibrate a small opportunity-cost estimator using only causal inputs.",
        "Rerun strict replay with the learned overlay; require Q1/Q3 nonnegative deltas under all deterministic stress levels.",
        "Keep Protocol101 as paper default; oracle overlay diagnostics are not live-deployable evidence.",
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
        "## Guardrail",
        "",
        payload["important_guardrail"],
        "",
        "## Oracle Overlay Diagnostic",
        "",
        table(
            payload["oracle_overlay_summary"],
            [
                "slippage_per_side",
                "split",
                "challenger_entries",
                "oracle_kept_entries",
                "original_delta_vs_protocol101",
                "oracle_overlay_delta_vs_protocol101",
                "blocked_protocol101_pnl_released",
            ],
        ),
        "",
        "## Causal Overlay Contract",
        "",
        f"- Decision rule: {payload['causal_overlay_contract']['decision_rule']}",
        f"- Promotion gate: {payload['causal_overlay_contract']['promotion_gate']}",
        "- Forbidden inputs:",
    ]
    lines.extend(f"  - {item}" for item in payload["causal_overlay_contract"]["forbidden_inputs"])
    lines.extend(["", "## Diagnosis", ""])
    lines.extend(f"- {item}" for item in payload["diagnosis"])
    lines.extend(["", "## Next Required Evidence", ""])
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
                values.append(f"{value:.2f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


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
