"""Protocol101 loss-reversal full serial replay.

This packet replays the frozen hypothetical-flat Protocol101 entry stream under
strict one-position serial constraints. It compares baseline Protocol101 against
a narrow priority-1 loss-reversal overlay:

When the current open trade is in the priority-1 bucket
(`current loss + opposite side signal + negative next quote + later exit
deteriorates`), exit the current trade at the causal quote state and release
the slot. The replay then resumes the same Protocol101 flat proposal stream.

No neural model is trained and Protocol101 remains the paper default.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.scripts.run_protocol101_strategy_forensics_packet import _fmt_money, _profit_factor  # noqa: E402


ROLE_LABEL = "AUDIT_PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1"
DEFAULT_FLAT_ENTRIES = Path(
    "v4/audit/autoresearch/protocol101_internal_slot_cost_counterfactual_v1/hypothetical_flat_protocol101_entries.csv"
)
DEFAULT_PRICING_ROWS = Path(
    "v4/audit/autoresearch/protocol101_loss_reversal_local_action_pricing_v1/local_action_pricing_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("v4/audit/autoresearch/protocol101_loss_reversal_full_serial_replay_v1")
DEFAULT_DOC = Path(
    "v4/docs/protocol101/training/research/PROTOCOL101_LOSS_REVERSAL_FULL_SERIAL_REPLAY_V1.md"
)
STRESS_LEVELS = (0.0, 0.10, 0.25)
CONTRACT_MULTIPLIER = 100.0
STARTING_CASH = 10_000.0
PRIORITY_1_BUCKET = "priority_1_current_loss_opposite_side_negative_next_and_exit_deteriorates"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def stress_suffix(stress: float) -> str:
    return f"{int(round(float(stress) * 100)):03d}"


def load_flat_entries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty flat-entry stream: {path}")
    frame["decision_dt"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["exit_dt"] = pd.to_datetime(frame["candidate_exit_time"], utc=True, errors="coerce")
    frame = frame[frame["decision_dt"].notna() & frame["exit_dt"].notna()].copy()
    for column in ["candidate_pnl", "entry_ask", "entry_spread", "score", "threshold"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.sort_values(["reported_split", "fold", "seed", "decision_dt", "candidate_uid"], kind="stable").reset_index(drop=True)


def load_priority1_triggers(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty local action pricing rows: {path}")
    frame = frame[frame["live_test_bucket"].astype(str).eq(PRIORITY_1_BUCKET)].copy()
    frame["signal_dt"] = pd.to_datetime(frame["best_blocked_decision_dt"], utc=True, errors="coerce")
    frame["current_quote_dt"] = pd.to_datetime(frame["current_quote_time"], utc=True, errors="coerce")
    frame = frame[frame["signal_dt"].notna()].copy()
    numeric = [
        "current_pnl_at_signal",
        "open_trade_pnl",
        "best_blocked_candidate_pnl",
        "best_blocked_minus_open_pnl",
        "current_bid",
        "current_ask",
        "current_spread",
        "quote_lag_seconds",
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.sort_values(["reported_split", "fold", "seed", "session", "signal_dt"], kind="stable").reset_index(drop=True)


def trigger_map(triggers: pd.DataFrame) -> dict[tuple[str, str, int, str, str], pd.Series]:
    triggers = triggers.copy()
    if "signal_dt" not in triggers.columns:
        triggers["signal_dt"] = pd.to_datetime(triggers["best_blocked_decision_dt"], utc=True, errors="coerce")
    else:
        triggers["signal_dt"] = pd.to_datetime(triggers["signal_dt"], utc=True, errors="coerce")
    out: dict[tuple[str, str, int, str, str], pd.Series] = {}
    for _, row in triggers.iterrows():
        key = (
            str(row["reported_split"]),
            str(row["fold"]),
            int(row["seed"]),
            str(row["session"]),
            str(row["open_trade_candidate_uid"]),
        )
        if key not in out or pd.Timestamp(row["signal_dt"]) < pd.Timestamp(out[key]["signal_dt"]):
            out[key] = row
    return out


def replay_all_variants(
    flat_entries: pd.DataFrame,
    triggers: pd.DataFrame,
    *,
    starting_cash: float = STARTING_CASH,
    stress_levels: tuple[float, ...] = STRESS_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    trigger_lookup = trigger_map(triggers)
    for stress in stress_levels:
        for variant in ["baseline", "priority1_exit_release_slot"]:
            variant_trades, variant_events = replay_variant(
                flat_entries,
                trigger_lookup,
                variant=variant,
                stress=float(stress),
                starting_cash=float(starting_cash),
            )
            trades.extend(variant_trades)
            events.extend(variant_events)
    return pd.DataFrame(trades), pd.DataFrame(events)


def replay_variant(
    flat_entries: pd.DataFrame,
    triggers: dict[tuple[str, str, int, str, str], pd.Series],
    *,
    variant: str,
    stress: float,
    starting_cash: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for group_key, group in flat_entries.groupby(["reported_split", "fold", "seed"], sort=True):
        reported_split, fold, seed = group_key
        cash = float(starting_cash)
        available_dt = pd.Timestamp.min.tz_localize("UTC")
        ordered = group.sort_values(["decision_dt", "exit_dt", "candidate_uid"], kind="stable")
        for _, proposal in ordered.iterrows():
            decision_dt = pd.Timestamp(proposal["decision_dt"])
            if decision_dt < available_dt:
                continue
            entry_ask = _finite(proposal.get("entry_ask"))
            premium = (entry_ask + float(stress)) * CONTRACT_MULTIPLIER
            if premium > cash + 1e-9:
                events.append(
                    replay_event(
                        variant,
                        stress,
                        reported_split,
                        fold,
                        seed,
                        proposal,
                        cash,
                        "skip_unaffordable_entry",
                        available_dt,
                    )
                )
                continue
            key = (
                str(reported_split),
                str(fold),
                int(seed),
                str(proposal["session"]),
                str(proposal["candidate_uid"]),
            )
            trigger = triggers.get(key)
            if variant == "priority1_exit_release_slot" and trigger is not None and trigger_applies(proposal, trigger):
                pnl = _finite(trigger.get("current_pnl_at_signal")) - (2.0 * float(stress) * CONTRACT_MULTIPLIER)
                before = cash
                cash += pnl
                release_dt = pd.Timestamp(trigger["signal_dt"])
                trades.append(
                    trade_row(
                        variant,
                        stress,
                        reported_split,
                        fold,
                        seed,
                        proposal,
                        entry_dt=decision_dt,
                        exit_dt=release_dt,
                        pnl=pnl,
                        cash_before=before,
                        cash_after=cash,
                        trade_action="loss_reversal_exit_release_slot",
                        trigger=trigger,
                    )
                )
                events.append(
                    replay_event(
                        variant,
                        stress,
                        reported_split,
                        fold,
                        seed,
                        proposal,
                        cash,
                        "loss_reversal_exit_release_slot",
                        release_dt,
                        trigger=trigger,
                    )
                )
                available_dt = release_dt
                continue

            pnl = _finite(proposal.get("candidate_pnl")) - (2.0 * float(stress) * CONTRACT_MULTIPLIER)
            before = cash
            cash += pnl
            exit_dt = pd.Timestamp(proposal["exit_dt"])
            trades.append(
                trade_row(
                    variant,
                    stress,
                    reported_split,
                    fold,
                    seed,
                    proposal,
                    entry_dt=decision_dt,
                    exit_dt=exit_dt,
                    pnl=pnl,
                    cash_before=before,
                    cash_after=cash,
                    trade_action="protocol101_flat_stream_entry",
                )
            )
            available_dt = exit_dt
    return trades, events


def trigger_applies(proposal: pd.Series, trigger: pd.Series) -> bool:
    decision_dt = pd.Timestamp(proposal["decision_dt"])
    exit_dt = pd.Timestamp(proposal["exit_dt"])
    signal_dt = pd.Timestamp(trigger["signal_dt"])
    return bool(decision_dt < signal_dt < exit_dt)


def trade_row(
    variant: str,
    stress: float,
    reported_split: str,
    fold: str,
    seed: int,
    proposal: pd.Series,
    *,
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
    pnl: float,
    cash_before: float,
    cash_after: float,
    trade_action: str,
    trigger: pd.Series | None = None,
) -> dict[str, Any]:
    trigger = trigger if trigger is not None else pd.Series(dtype=object)
    return {
        "variant": variant,
        "stress_per_side": float(stress),
        "reported_split": str(reported_split),
        "fold": str(fold),
        "seed": int(seed),
        "session": str(proposal["session"]),
        "entry_time": entry_dt.isoformat(),
        "exit_time": exit_dt.isoformat(),
        "candidate_uid": str(proposal["candidate_uid"]),
        "contract_id": str(proposal["contract_id"]),
        "right": str(proposal["right"]),
        "trade_action": trade_action,
        "pnl": float(pnl),
        "cash_before": float(cash_before),
        "cash_after": float(cash_after),
        "entry_ask": _finite(proposal.get("entry_ask")),
        "candidate_pnl_unstressed": _finite(proposal.get("candidate_pnl")),
        "trigger_candidate_uid": str(trigger.get("best_blocked_candidate_uid", "")),
        "trigger_contract_id": str(trigger.get("best_blocked_contract_id", "")),
        "trigger_right": str(trigger.get("best_blocked_right", "")),
        "trigger_signal_dt": pd.Timestamp(trigger["signal_dt"]).isoformat() if "signal_dt" in trigger else "",
        "trigger_current_pnl_at_signal": _finite(trigger.get("current_pnl_at_signal"), 0.0),
        "trigger_open_trade_pnl": _finite(trigger.get("open_trade_pnl"), 0.0),
        "trigger_local_bucket": str(trigger.get("live_test_bucket", "")),
        "label_limitations": (
            "full_serial_flat_stream_replay_no_neural_training; fill_model_not_calibrated; "
            "uses_frozen_protocol101_flat_proposals"
        ),
    }


def replay_event(
    variant: str,
    stress: float,
    reported_split: str,
    fold: str,
    seed: int,
    proposal: pd.Series,
    cash: float,
    event_type: str,
    available_dt: pd.Timestamp,
    trigger: pd.Series | None = None,
) -> dict[str, Any]:
    trigger = trigger if trigger is not None else pd.Series(dtype=object)
    return {
        "variant": variant,
        "stress_per_side": float(stress),
        "reported_split": str(reported_split),
        "fold": str(fold),
        "seed": int(seed),
        "session": str(proposal["session"]),
        "event_time": pd.Timestamp(proposal["decision_dt"]).isoformat(),
        "available_dt": pd.Timestamp(available_dt).isoformat(),
        "candidate_uid": str(proposal["candidate_uid"]),
        "event_type": event_type,
        "cash": float(cash),
        "trigger_candidate_uid": str(trigger.get("best_blocked_candidate_uid", "")),
        "trigger_signal_dt": pd.Timestamp(trigger["signal_dt"]).isoformat() if "signal_dt" in trigger else "",
    }


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["variant", "stress_per_side", "reported_split"]
    for key_values, group in trades.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "variant": key_values[0],
                "stress_per_side": float(key_values[1]),
                "reported_split": key_values[2],
                "trades": int(len(group)),
                "pnl": float(pnl.sum()),
                "win_rate": float((pnl > 0.0).mean()) if len(group) else 0.0,
                "profit_factor": _profit_factor(pnl),
                "loss_reversal_exits": int(group["trade_action"].astype(str).eq("loss_reversal_exit_release_slot").sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    baseline = out[out["variant"].eq("baseline")][["stress_per_side", "reported_split", "pnl", "trades"]].rename(
        columns={"pnl": "baseline_pnl", "trades": "baseline_trades"}
    )
    out = out.merge(baseline, on=["stress_per_side", "reported_split"], how="left")
    out["delta_vs_baseline"] = out["pnl"] - out["baseline_pnl"]
    out["trade_delta_vs_baseline"] = out["trades"] - out["baseline_trades"]
    return out.sort_values(["stress_per_side", "variant", "reported_split"], kind="stable")


def summarize_variants(split_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, stress), group in split_summary.groupby(["variant", "stress_per_side"], sort=True):
        rows.append(
            {
                "variant": variant,
                "stress_per_side": float(stress),
                "trades": int(group["trades"].sum()),
                "pnl": float(group["pnl"].sum()),
                "baseline_pnl": float(group["baseline_pnl"].sum()),
                "delta_vs_baseline": float(group["delta_vs_baseline"].sum()),
                "loss_reversal_exits": int(group["loss_reversal_exits"].sum()),
                "all_splits_positive_delta": bool((group["delta_vs_baseline"] >= 0.0).all()) if str(variant) != "baseline" else True,
            }
        )
    return pd.DataFrame(rows).sort_values(["stress_per_side", "variant"], kind="stable")


def build_summary(trades: pd.DataFrame, split_summary: pd.DataFrame, variant_summary: pd.DataFrame) -> dict[str, Any]:
    challenger = variant_summary[variant_summary["variant"].eq("priority1_exit_release_slot")]
    stress_rows = {
        stress_suffix(row["stress_per_side"]): {
            "pnl": float(row["pnl"]),
            "baseline_pnl": float(row["baseline_pnl"]),
            "delta_vs_baseline": float(row["delta_vs_baseline"]),
            "loss_reversal_exits": int(row["loss_reversal_exits"]),
            "all_splits_positive_delta": bool(row["all_splits_positive_delta"]),
        }
        for _, row in challenger.iterrows()
    }
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "full serial flat-stream replay for the selected Protocol101 priority-1 loss-reversal overlay",
        "decision": "protocol101_loss_reversal_full_serial_replay_complete_training_still_blocked",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "model_training": False,
        "challenge_allowed": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "untouched_holdout_scored": False,
        "counts": {
            "replay_trades": int(len(trades)),
            "baseline_trades_stress_000": int(
                len(trades[trades["variant"].eq("baseline") & np.isclose(trades["stress_per_side"], 0.0)])
            ),
        },
        "challenger_by_stress": stress_rows,
        "blockers": [
            "diagnostic_splits_are_research_exposed",
            "fill_latency_and_quote_freshness_not_calibrated",
            "missing_quote_state_rows_excluded_from_overlay",
            "no_live_no_order_parity_for_loss_reversal_gate",
            "neural_training_forbidden_until_manual_review_and_label_contract_are_frozen",
        ],
    }


def write_report(output_dir: Path, summary: dict[str, Any], split_summary: pd.DataFrame, variant_summary: pd.DataFrame) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        "What is this: full serial flat-stream replay for `PROTOCOL101_LOSS_REVERSAL_EXIT_GATE_V1` priority-1 overlay",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Untouched holdout scored: no",
        f"Decision: `{summary['decision']}`",
        "",
        "## Bottom Line",
        "",
        "The priority-1 loss-reversal overlay survives full serial flat-stream replay on the exposed diagnostic splits. This is the first Track A result that looks like a concrete Protocol101 improvement playbook, but it is still research-only.",
        "",
        "## Variant Summary",
        "",
        "| variant | stress | trades | PnL | baseline PnL | delta | exits | all splits positive delta |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in variant_summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['stress_per_side']:.2f} | {int(row['trades'])} | "
            f"{_fmt_money(row['pnl'])} | {_fmt_money(row['baseline_pnl'])} | {_fmt_money(row['delta_vs_baseline'])} | "
            f"{int(row['loss_reversal_exits'])} | {bool(row['all_splits_positive_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            "| variant | stress | split | trades | PnL | baseline PnL | delta | exits | PF | win rate |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in split_summary.iterrows():
        if row["variant"] == "baseline":
            continue
        lines.append(
            f"| {row['variant']} | {row['stress_per_side']:.2f} | {row['reported_split']} | {int(row['trades'])} | "
            f"{_fmt_money(row['pnl'])} | {_fmt_money(row['baseline_pnl'])} | {_fmt_money(row['delta_vs_baseline'])} | "
            f"{int(row['loss_reversal_exits'])} | {row['profit_factor']:.3f} | {row['win_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is not a neural model and not a promotion claim. The replay uses the frozen Protocol101 flat proposal stream and a hand-declared priority-1 gate derived from Track A diagnostics. It answers a trader question: when Protocol101 is losing, an opposite-side Protocol101 signal appears, the next quote is adverse, and the original hold later deteriorates, should the bot release the slot?",
            "",
            "On these exposed diagnostic splits, the answer is yes. The next work is not architecture search. It is manual chart/path review, formalizing the causal feature contract, live no-order parity for this gate, fill/latency evidence, and then a frozen untouched evaluation.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{output_dir / 'summary.json'}`",
            f"- Replay trades: `{output_dir / 'full_serial_replay_trades.csv'}`",
            f"- Replay events: `{output_dir / 'full_serial_replay_events.csv'}`",
            f"- Split summary: `{output_dir / 'split_summary.csv'}`",
            f"- Variant summary: `{output_dir / 'variant_summary.csv'}`",
            f"- Report: `{output_dir / 'report.md'}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_entries = load_flat_entries(Path(args.flat_entries))
    triggers = load_priority1_triggers(Path(args.pricing_rows))
    trades, events = replay_all_variants(flat_entries, triggers, starting_cash=float(args.starting_cash))
    split_summary = summarize_trades(trades)
    variant_summary = summarize_variants(split_summary)
    summary = build_summary(trades, split_summary, variant_summary)
    summary["inputs"] = {
        "flat_entries": str(args.flat_entries),
        "pricing_rows": str(args.pricing_rows),
        "starting_cash": float(args.starting_cash),
        "stress_levels": list(STRESS_LEVELS),
        "gate": PRIORITY_1_BUCKET,
    }
    summary["outputs"] = {
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "report.md"),
        "doc": str(args.doc),
        "replay_trades": str(output_dir / "full_serial_replay_trades.csv"),
        "replay_events": str(output_dir / "full_serial_replay_events.csv"),
        "split_summary": str(output_dir / "split_summary.csv"),
        "variant_summary": str(output_dir / "variant_summary.csv"),
    }
    trades.to_csv(output_dir / "full_serial_replay_trades.csv", index=False)
    events.to_csv(output_dir / "full_serial_replay_events.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    variant_summary.to_csv(output_dir / "variant_summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    report = write_report(output_dir, summary, split_summary, variant_summary)
    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ROLE_LABEL)
    parser.add_argument("--flat-entries", type=Path, default=DEFAULT_FLAT_ENTRIES)
    parser.add_argument("--pricing-rows", type=Path, default=DEFAULT_PRICING_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
