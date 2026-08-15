"""Protocol101 internal slot-cost counterfactual audit.

This is a research-only Track A diagnostic. It scores the frozen Protocol101
event policy at every candidate event as if the account were flat, then asks
which model-approved Protocol101 entries were unavailable because the actual
strict-serial Protocol101 policy was already holding one contract.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.supervised_pilot import FeatureScaler  # noqa: E402
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS  # noqa: E402
from v4.scripts.run_protocol097_sequential_event_policy import EventPolicyConfig, EventSetPolicy, build_events, predict_event_action  # noqa: E402
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS, add_causal_history_features  # noqa: E402
from v4.scripts.run_protocol101_strategy_forensics_packet import _fmt_money  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_INTERNAL_SLOT_COST_COUNTERFACTUAL_V1"
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_INTERNAL_SLOT_COST_COUNTERFACTUAL_V1.md"
)
DEFAULT_PROTOCOL092_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy/serial_opportunity_dataset.parquet")
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_ACTUAL_TRADES = DEFAULT_PROTOCOL101_DIR / "serial_policy_trades.json"


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    dataset = pd.read_parquet(path)
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True, errors="coerce")
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True, errors="coerce")
    dataset = dataset[dataset["decision_dt"].notna() & dataset["candidate_exit_dt"].notna()].copy()
    events = build_events(dataset)
    add_causal_history_features(events)
    return events


def load_actual_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.DataFrame(json.loads(path.read_text()))
    if frame.empty:
        return frame
    frame["entry_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_dt"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["pnl"] = pd.to_numeric(frame["pnl"], errors="coerce").fillna(0.0)
    return frame[frame["entry_dt"].notna() & frame["exit_dt"].notna()].reset_index(drop=True)


def load_policy(protocol101_dir: Path, fold_name: str, seed: int) -> tuple[EventSetPolicy, FeatureScaler, float]:
    model_dir = protocol101_dir / "model_artifacts" / fold_name / f"seed_{seed}"
    manifest_path = model_dir / "manifest.json"
    model_path = model_dir / "model.pt"
    scaler_path = model_dir / "scaler.json"
    summary_path = protocol101_dir / "summary.json"
    for path in [manifest_path, model_path, scaler_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 96))
    model = EventSetPolicy(input_dim=len(FEATURE_COLUMNS), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    scaler_payload = json.loads(scaler_path.read_text())
    scaler = FeatureScaler(
        fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
        mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        std=np.asarray(scaler_payload["std"], dtype=np.float32),
    )
    threshold = protocol101_threshold(summary_path, fold_name=fold_name, seed=seed)
    return model, scaler, threshold


def protocol101_threshold(summary_path: Path, *, fold_name: str, seed: int) -> float:
    summary = json.loads(summary_path.read_text())
    for row in summary.get("fold_results", []):
        if str(row.get("fold")) == str(fold_name) and int(row.get("seed", -1)) == int(seed):
            return float(row["threshold"])
    raise ValueError(f"missing threshold for fold={fold_name} seed={seed}")


def event_slices_for_fold(events: list[dict[str, Any]], fold: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    test = [event for event in events if event["split"] == fold["test_split"] and int(event["seed"]) == int(seed)]
    out = {str(fold["test_split"]): test}
    if fold["test_split"] == "q1_2026":
        out["march_2026"] = [event for event in test if str(event["session"]) >= "2026-03-01"]
    return out


def score_hypothetical_flat_entries(
    events: list[dict[str, Any]],
    *,
    protocol101_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config = EventPolicyConfig()
    for fold in FOLDS:
        fold_name = str(fold["name"])
        for seed in [1, 2, 3, 4, 5]:
            model, scaler, threshold = load_policy(protocol101_dir, fold_name, seed)
            for reported_split, split_events in event_slices_for_fold(events, fold, seed).items():
                for event in sorted(split_events, key=lambda item: item["decision_dt"]):
                    action, margin = predict_event_action(event, model, scaler, feature_columns=FEATURE_COLUMNS)
                    if action <= 0 or margin < threshold:
                        continue
                    row = event["candidates"].iloc[action - 1]
                    rows.append(
                        {
                            "fold": fold_name,
                            "reported_split": reported_split,
                            "split": str(row["split"]),
                            "seed": int(seed),
                            "session": str(row["session"]),
                            "decision_time": pd.Timestamp(row["decision_dt"]).isoformat(),
                            "decision_dt": pd.Timestamp(row["decision_dt"]),
                            "candidate_exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
                            "candidate_uid": str(row["candidate_uid"]),
                            "trade_uid": str(row["trade_uid"]),
                            "contract_id": str(row["contract_id"]),
                            "right": str(row["right"]),
                            "offset": finite(row.get("offset")),
                            "score": float(margin),
                            "threshold": float(threshold),
                            "candidate_pnl": finite(row.get("candidate_pnl")),
                            "entry_ask": finite(row.get("entry_ask")),
                            "entry_spread": finite(row.get("entry_spread")),
                            "edge": finite(row.get("edge")),
                            "time_bucket": str(row.get("time_bucket", "")),
                            "feature_columns": ",".join(FEATURE_COLUMNS),
                            "model_config": json.dumps(config.__dict__, sort_keys=True),
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["reported_split", "fold", "seed", "session", "decision_dt"], kind="stable").reset_index(drop=True)


def attribute_blocked_entries(hypothetical: pd.DataFrame, actual_trades: pd.DataFrame) -> pd.DataFrame:
    if hypothetical.empty or actual_trades.empty:
        return pd.DataFrame()
    hypo = hypothetical.copy()
    hypo["decision_dt"] = pd.to_datetime(hypo["decision_dt"], utc=True, errors="coerce")
    actual = actual_trades.copy()
    actual["entry_dt"] = pd.to_datetime(actual["entry_dt"], utc=True, errors="coerce")
    actual["exit_dt"] = pd.to_datetime(actual["exit_dt"], utc=True, errors="coerce")
    blocked: list[dict[str, Any]] = []
    for key, group in hypo.groupby(["reported_split", "fold", "seed", "session"], sort=False):
        split, fold, seed, session = key
        intervals = actual[
            actual["reported_split"].astype(str).eq(str(split))
            & actual["fold"].astype(str).eq(str(fold))
            & actual["seed"].astype(int).eq(int(seed))
            & actual["session"].astype(str).eq(str(session))
        ].sort_values(["entry_dt", "exit_dt"])
        if intervals.empty:
            continue
        for _, candidate in group.iterrows():
            dt = pd.Timestamp(candidate["decision_dt"])
            containing = intervals[(intervals["entry_dt"] < dt) & (dt < intervals["exit_dt"])]
            if containing.empty:
                continue
            open_trade = containing.iloc[0]
            blocked.append(blocked_row(candidate, open_trade))
    out = pd.DataFrame(blocked)
    if out.empty:
        return out
    return out.sort_values(["reported_split", "fold", "seed", "session", "blocked_decision_dt"], kind="stable").reset_index(drop=True)


def blocked_row(candidate: pd.Series, open_trade: pd.Series) -> dict[str, Any]:
    blocked_pnl = finite(candidate.get("candidate_pnl"))
    open_pnl = finite(open_trade.get("pnl"))
    return {
        "reported_split": str(candidate["reported_split"]),
        "fold": str(candidate["fold"]),
        "seed": int(candidate["seed"]),
        "session": str(candidate["session"]),
        "blocked_decision_dt": pd.Timestamp(candidate["decision_dt"]).isoformat(),
        "blocked_candidate_uid": str(candidate["candidate_uid"]),
        "blocked_contract_id": str(candidate["contract_id"]),
        "blocked_right": str(candidate["right"]),
        "blocked_offset": finite(candidate.get("offset")),
        "blocked_score": finite(candidate.get("score")),
        "blocked_threshold": finite(candidate.get("threshold")),
        "blocked_candidate_pnl": blocked_pnl,
        "blocked_entry_ask": finite(candidate.get("entry_ask")),
        "blocked_entry_spread": finite(candidate.get("entry_spread")),
        "blocked_time_bucket": str(candidate.get("time_bucket", "")),
        "open_trade_entry_dt": pd.Timestamp(open_trade["entry_dt"]).isoformat(),
        "open_trade_exit_dt": pd.Timestamp(open_trade["exit_dt"]).isoformat(),
        "open_trade_candidate_uid": str(open_trade["candidate_uid"]),
        "open_trade_contract_id": str(open_trade["contract_id"]),
        "open_trade_right": str(open_trade["right"]),
        "open_trade_pnl": open_pnl,
        "open_trade_exit_reason": str(open_trade.get("exit_reason", "")),
        "minutes_after_open_entry": float((pd.Timestamp(candidate["decision_dt"]) - pd.Timestamp(open_trade["entry_dt"])).total_seconds() / 60.0),
        "minutes_before_open_exit": float((pd.Timestamp(open_trade["exit_dt"]) - pd.Timestamp(candidate["decision_dt"])).total_seconds() / 60.0),
        "open_minus_blocked_pnl": open_pnl - blocked_pnl,
        "label_limitations": "blocked_events_are_counterfactual_flat_upper_bound_not_additive_replay",
    }


def build_open_trade_slot_summary(blocked: pd.DataFrame) -> pd.DataFrame:
    if blocked.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = [
        "reported_split",
        "fold",
        "seed",
        "session",
        "open_trade_candidate_uid",
        "open_trade_contract_id",
        "open_trade_entry_dt",
        "open_trade_exit_dt",
        "open_trade_right",
        "open_trade_exit_reason",
    ]
    for key_values, group in blocked.groupby(keys, dropna=False, sort=False):
        candidate_pnl = pd.to_numeric(group["blocked_candidate_pnl"], errors="coerce").fillna(0.0)
        best_idx = int(candidate_pnl.idxmax())
        best = group.loc[best_idx]
        open_pnl = finite(best.get("open_trade_pnl"))
        rows.append(
            {
                **{column: value for column, value in zip(keys, key_values)},
                "blocked_entry_events": int(len(group)),
                "positive_blocked_entry_events": int((candidate_pnl > 0.0).sum()),
                "sum_blocked_candidate_pnl_non_additive": float(candidate_pnl.sum()),
                "best_blocked_candidate_pnl": float(candidate_pnl.max()),
                "best_blocked_candidate_uid": str(best["blocked_candidate_uid"]),
                "best_blocked_decision_dt": str(best["blocked_decision_dt"]),
                "open_trade_pnl": open_pnl,
                "open_minus_best_blocked_pnl": open_pnl - float(candidate_pnl.max()),
                "slot_cost_positive_vs_best_blocked": bool(float(candidate_pnl.max()) > open_pnl),
            }
        )
    return pd.DataFrame(rows).sort_values(["reported_split", "fold", "seed", "session", "open_trade_entry_dt"], kind="stable")


def build_split_summary(hypothetical: pd.DataFrame, blocked: pd.DataFrame, open_slot: pd.DataFrame) -> pd.DataFrame:
    splits = sorted(set(hypothetical.get("reported_split", pd.Series(dtype=str)).astype(str)) | set(blocked.get("reported_split", pd.Series(dtype=str)).astype(str)))
    rows: list[dict[str, Any]] = []
    for split in splits:
        hyp = hypothetical[hypothetical["reported_split"].astype(str).eq(split)] if not hypothetical.empty else pd.DataFrame()
        blk = blocked[blocked["reported_split"].astype(str).eq(split)] if not blocked.empty else pd.DataFrame()
        slot = open_slot[open_slot["reported_split"].astype(str).eq(split)] if not open_slot.empty else pd.DataFrame()
        rows.append(
            {
                "reported_split": split,
                "hypothetical_flat_entries": int(len(hyp)),
                "hypothetical_flat_pnl": finite_sum(hyp.get("candidate_pnl", pd.Series(dtype=float))),
                "blocked_entry_events": int(len(blk)),
                "blocked_candidate_pnl_non_additive": finite_sum(blk.get("blocked_candidate_pnl", pd.Series(dtype=float))),
                "positive_blocked_entry_events": int((pd.to_numeric(blk.get("blocked_candidate_pnl", pd.Series(dtype=float)), errors="coerce") > 0.0).sum()),
                "open_trades_with_blocked_entries": int(len(slot)),
                "open_trades_where_best_blocked_beats_open": int(slot.get("slot_cost_positive_vs_best_blocked", pd.Series(dtype=bool)).astype(bool).sum())
                if not slot.empty
                else 0,
                "best_blocked_minus_open_total": float((-pd.to_numeric(slot.get("open_minus_best_blocked_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)).sum())
                if not slot.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("reported_split", kind="stable")


def build_summary(
    hypothetical: pd.DataFrame,
    blocked: pd.DataFrame,
    open_slot: pd.DataFrame,
    split_summary: pd.DataFrame,
    *,
    protocol092_dataset: Path,
    protocol101_dir: Path,
    actual_trades: Path,
) -> dict[str, Any]:
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "Track A Protocol101 internal slot-cost counterfactual / flat-mode event replay",
        "decision": "protocol101_internal_slot_cost_counterfactual_complete_training_still_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "inputs": {
            "protocol092_dataset": str(protocol092_dataset),
            "protocol101_dir": str(protocol101_dir),
            "actual_trades": str(actual_trades),
        },
        "counts": {
            "hypothetical_flat_entries": int(len(hypothetical)),
            "blocked_entry_events": int(len(blocked)),
            "open_trades_with_blocked_entries": int(len(open_slot)),
            "open_trades_where_best_blocked_beats_open": int(open_slot.get("slot_cost_positive_vs_best_blocked", pd.Series(dtype=bool)).astype(bool).sum())
            if not open_slot.empty
            else 0,
        },
        "aggregate": {
            "hypothetical_flat_pnl": finite_sum(hypothetical.get("candidate_pnl", pd.Series(dtype=float))),
            "blocked_candidate_pnl_non_additive": finite_sum(blocked.get("blocked_candidate_pnl", pd.Series(dtype=float))),
            "best_blocked_minus_open_total": float((-pd.to_numeric(open_slot.get("open_minus_best_blocked_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)).sum())
            if not open_slot.empty
            else 0.0,
        },
        "split_summary": split_summary.to_dict(orient="records"),
        "interpretation": [
            "This is not a tradable replay because multiple blocked flat entries inside one open interval are mutually exclusive.",
            "Use open_trade_slot_summary as the less-overcounted view: one best blocked candidate per actual open Protocol101 trade.",
            "Positive blocked-candidate value means Protocol101 may have spent the slot on a weaker earlier trade; it does not prove the later trade was live-fillable.",
        ],
        "blockers": [
            "fill_latency_and_quote_freshness_not_calibrated",
            "blocked_events_are_counterfactual_flat_upper_bound_not_additive_replay",
            "no_untouched_holdout_scored",
            "no_model_training_authorized",
        ],
        "challenge_allowed": False,
        "training_allowed": False,
    }


def write_report(output_dir: Path, summary: dict[str, Any], split_summary: pd.DataFrame) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: Protocol101 internal slot-cost counterfactual",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "This packet closes the previous Track A blocker by scoring the frozen Protocol101 event policy at every event as if the account were flat. It then attributes model-approved entries that actual strict-serial Protocol101 could not take because one contract was already open.",
        "",
        "This is evidence about Protocol101's internal slot opportunity cost, not a replacement strategy. Blocked events inside a single open interval are mutually exclusive, and the audit has no calibrated live fill model.",
        "",
        "## Headline",
        "",
        f"- Hypothetical flat Protocol101 entries: `{summary['counts']['hypothetical_flat_entries']}`",
        f"- Blocked entry events while actual Protocol101 was holding: `{summary['counts']['blocked_entry_events']}`",
        f"- Actual open trades with at least one blocked entry: `{summary['counts']['open_trades_with_blocked_entries']}`",
        f"- Open trades where best blocked entry beats actual open-trade PnL: `{summary['counts']['open_trades_where_best_blocked_beats_open']}`",
        f"- Non-additive blocked candidate PnL: `{_fmt_money(summary['aggregate']['blocked_candidate_pnl_non_additive'])}`",
        f"- Best-blocked-minus-open total: `{_fmt_money(summary['aggregate']['best_blocked_minus_open_total'])}`",
        "",
        "## Split Summary",
        "",
        "| split | flat entries | blocked events | positive blocked | open trades w/ blocked | best-blocked beats open | best-blocked-minus-open |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"| {row['reported_split']} | {int(row['hypothetical_flat_entries'])} | {int(row['blocked_entry_events'])} | "
            f"{int(row['positive_blocked_entry_events'])} | {int(row['open_trades_with_blocked_entries'])} | "
            f"{int(row['open_trades_where_best_blocked_beats_open'])} | {_fmt_money(row['best_blocked_minus_open_total'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If this audit shows large positive best-blocked-minus-open values, Protocol101 may be taking some early trades that consume the only slot before stronger later Protocol101 signals.",
            "- If it is small or concentrated in one split, the internal slot-cost hypothesis is weaker than the runner/giveback and hard-stop questions.",
            "- This must not be used as a training label until fill/latency realism and a mutually exclusive counterfactual replay are added.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Hypothetical flat entries: `{output_dir / 'hypothetical_flat_protocol101_entries.csv'}`",
            f"- Blocked events: `{output_dir / 'blocked_protocol101_internal_slot_events.csv'}`",
            f"- Open trade slot summary: `{output_dir / 'open_trade_slot_summary.csv'}`",
            f"- Split summary: `{output_dir / 'split_summary.csv'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_sum(values: Any) -> float:
    if values is None:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(Path(args.protocol092_dataset))
    actual = load_actual_trades(Path(args.actual_trades))
    hypothetical = score_hypothetical_flat_entries(events, protocol101_dir=Path(args.protocol101_dir))
    blocked = attribute_blocked_entries(hypothetical, actual)
    open_slot = build_open_trade_slot_summary(blocked)
    split_summary = build_split_summary(hypothetical, blocked, open_slot)

    hypothetical.to_csv(output_dir / "hypothetical_flat_protocol101_entries.csv", index=False)
    blocked.to_csv(output_dir / "blocked_protocol101_internal_slot_events.csv", index=False)
    open_slot.to_csv(output_dir / "open_trade_slot_summary.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    summary = build_summary(
        hypothetical,
        blocked,
        open_slot,
        split_summary,
        protocol092_dataset=Path(args.protocol092_dataset),
        protocol101_dir=Path(args.protocol101_dir),
        actual_trades=Path(args.actual_trades),
    )
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "hypothetical_flat_entries": str(output_dir / "hypothetical_flat_protocol101_entries.csv"),
        "blocked_events": str(output_dir / "blocked_protocol101_internal_slot_events.csv"),
        "open_trade_slot_summary": str(output_dir / "open_trade_slot_summary.csv"),
        "split_summary": str(output_dir / "split_summary.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, split_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--protocol092-dataset", type=Path, default=DEFAULT_PROTOCOL092_DATASET)
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--actual-trades", type=Path, default=DEFAULT_ACTUAL_TRADES)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
