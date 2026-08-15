"""AUDIT_MODEL_SELECTION_OVERFIT_RISK_V1.

Inventory v4 model-changing protocols and mark which evaluation blocks have
been repeatedly inspected. This is a meta-validation audit: it prevents us from
treating Q3/Q4/Q1/recent as sacred holdouts after many research loops have
already looked at them.

No paid data is downloaded. No broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "AUDIT_MODEL_SELECTION_OVERFIT_RISK_V1"
HISTORICAL_ID = "Protocol273"
DEFAULT_AUDIT_ROOT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_273_model_selection_overfit_risk")
DEFAULT_PROTOCOL265_TRADES = Path("v4/audit/autoresearch/v4_aplus_hypothesis_265_source_penalty_baseline_anchored_continuation/source_penalty_baseline_anchored_continuation_trades.csv")
SPLIT_NAMES = ["q4_2024_external", "q1_2025", "q2_2025", "q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--protocol265-trades", type=Path, default=DEFAULT_PROTOCOL265_TRADES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_summaries(args.audit_root)
    inventory.to_csv(args.out_dir / "protocol_inventory.csv", index=False)
    split_exposure = summarize_split_exposure(inventory)
    split_exposure.to_csv(args.out_dir / "split_exposure.csv", index=False)
    bootstrap = bootstrap_protocol265(args.protocol265_trades, reps=int(args.bootstrap_reps))
    bootstrap.to_csv(args.out_dir / "protocol265_day_bootstrap.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / model-selection overfit and false-discovery risk audit",
        "changes_paper_default": False,
        "candidate_label": "ALL_RESEARCH_CHALLENGERS",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "data_used": str(args.audit_root),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "row_counts": {
            "summaries_found": int(len(inventory)),
            "model_or_training_protocols": int(inventory["is_model_changing"].sum()) if not inventory.empty else 0,
            "bootstrap_rows": int(len(bootstrap)),
        },
        "split_exposure": split_exposure.to_dict("records"),
        "protocol265_day_bootstrap": bootstrap.to_dict("records"),
        "pbo_cscv_status": "blocked_requires_formal_strategy_matrix_before_final_promotion",
        "decision": decide(split_exposure),
        "next_experiment": "Reserve a new untouched block before any final promotion claim; current repeated-research blocks remain useful but not sacred.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def inventory_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        text = json.dumps(payload, default=str).lower()
        role = str(payload.get("role_label", payload.get("protocol", path.parent.name)))
        decision = str(payload.get("decision", ""))
        what = str(payload.get("what_is_this", ""))
        is_model = any(token in text for token in ["model_training", "candidate_label", "threshold", "epochs", "model_artifacts"])
        is_runtime = any(token in text for token in ["runtime", "no-order", "no_order", "broker_endpoint"])
        touched = [split for split in SPLIT_NAMES if split.lower() in text]
        rows.append(
            {
                "audit_dir": str(path.parent),
                "summary": str(path),
                "role_label": role,
                "what_is_this": what,
                "decision": decision,
                "is_model_changing": bool(is_model and "diagnostic" not in what.lower() and "dataset" not in what.lower()),
                "is_runtime_or_parity": bool(is_runtime),
                "splits_mentioned": ",".join(touched),
                **{f"mentions_{split}": split in touched for split in SPLIT_NAMES},
            }
        )
    return pd.DataFrame(rows)


def summarize_split_exposure(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if inventory.empty:
        return pd.DataFrame(columns=["split", "summaries_mentioning_split", "model_protocols_mentioning_split", "classification"])
    for split in SPLIT_NAMES:
        mentions = inventory[f"mentions_{split}"].astype(bool) if f"mentions_{split}" in inventory.columns else pd.Series(False, index=inventory.index)
        model_mentions = mentions & inventory["is_model_changing"].astype(bool)
        count = int(mentions.sum())
        model_count = int(model_mentions.sum())
        if split in {"q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"} and count >= 5:
            classification = "repeated_research_test_not_sacred_holdout"
        elif count == 0:
            classification = "untouched_or_not_inventory_visible"
        else:
            classification = "limited_research_exposure"
        rows.append(
            {
                "split": split,
                "summaries_mentioning_split": count,
                "model_protocols_mentioning_split": model_count,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_protocol265(path: Path, *, reps: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["reported_split", "days", "trades", "total_pnl", "bootstrap_mean", "ci05", "ci50", "ci95", "prob_positive"])
    frame = pd.read_csv(path)
    if "pnl" not in frame.columns and "candidate_pnl" in frame.columns:
        frame["pnl"] = frame["candidate_pnl"]
    for column in ["reported_split", "session", "pnl"]:
        if column not in frame.columns:
            return pd.DataFrame(columns=["reported_split", "days", "trades", "total_pnl", "bootstrap_mean", "ci05", "ci50", "ci95", "prob_positive"])
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    rng = np.random.default_rng(273)
    rows = []
    for split, group in frame.groupby("reported_split", sort=True):
        daily = group.groupby("session")["pnl"].sum().to_numpy(dtype=float)
        if len(daily) == 0:
            continue
        sims = []
        for _ in range(max(1, int(reps))):
            sample = rng.choice(daily, size=len(daily), replace=True)
            sims.append(float(sample.sum()))
        sims_arr = np.asarray(sims, dtype=float)
        rows.append(
            {
                "reported_split": str(split),
                "days": int(len(daily)),
                "trades": int(len(group)),
                "total_pnl": float(daily.sum()),
                "bootstrap_mean": float(sims_arr.mean()),
                "ci05": float(np.percentile(sims_arr, 5)),
                "ci50": float(np.percentile(sims_arr, 50)),
                "ci95": float(np.percentile(sims_arr, 95)),
                "prob_positive": float((sims_arr > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def decide(split_exposure: pd.DataFrame) -> str:
    if split_exposure.empty:
        return "blocked_no_audit_inventory"
    repeated = split_exposure["classification"].astype(str).eq("repeated_research_test_not_sacred_holdout").sum()
    if repeated > 0:
        return "model_selection_overfit_risk_confirmed_reserve_new_untouched_block"
    return "model_selection_overfit_risk_low_in_current_inventory"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Split Exposure",
        "",
        "| split | summaries | model protocols | classification |",
        "|---|---:|---:|---|",
    ]
    for row in payload["split_exposure"]:
        lines.append(f"| {row['split']} | {row['summaries_mentioning_split']} | {row['model_protocols_mentioning_split']} | {row['classification']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Q3/Q4/Q1/March/recent remain useful repeated-research tests, but they are no longer sacred untouched holdouts.",
            "A final promotion claim needs a newly reserved block that is not used during hypothesis search.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
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
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
