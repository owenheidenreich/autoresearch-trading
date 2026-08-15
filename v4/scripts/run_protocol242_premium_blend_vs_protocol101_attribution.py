"""AUDIT_PREMIUM_LEANING_BLEND_VS_PROTOCOL101_V1.

Historically Protocol242. This attribution audit compares the frozen research
challenger CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1 against
PAPER_DEFAULT_PROTOCOL101 under the same one-account serial replay framing.

It explains side exposure, time buckets, exact trade overlap, churn/re-entry,
and directional move capture. It does not change the paper default, download
market data, place orders, or call broker endpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import v4.scripts.run_protocol216_full_action_history_vs_protocol101_attribution as p216


ROLE_LABEL = "AUDIT_PREMIUM_LEANING_BLEND_VS_PROTOCOL101_V1"
HISTORICAL_ID = "Protocol242"
DEFAULT_CHALLENGER_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_240_premium_leaning_blend_five_seed_decision/five_seed_model_trades.csv"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_242_premium_blend_vs_protocol101_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-trades", type=Path, default=DEFAULT_CHALLENGER_TRADES)
    parser.add_argument("--protocol101-trades", type=Path, default=p216.DEFAULT_PROTOCOL101_TRADES)
    parser.add_argument("--recent-protocol101-trades", type=Path, default=p216.DEFAULT_RECENT_PROTOCOL101_TRADES)
    parser.add_argument("--normalized-dir", type=Path, default=p216.DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reentry-gap-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    challenger = p216.load_challenger(args.challenger_trades)
    protocol101 = p216.load_protocol101(args.protocol101_trades, args.recent_protocol101_trades)
    combined = pd.concat([challenger, protocol101], ignore_index=True)
    enriched, quote_skips = p216.enrich_from_normalized(combined, args.normalized_dir)

    side_summary = p216.summarize_group(enriched, ["reported_split", "policy", "right"])
    time_summary = p216.summarize_group(enriched, ["reported_split", "policy", "time_bucket"])
    exit_summary = p216.summarize_group(enriched, ["reported_split", "policy", "exit_reason"])
    move_summary = p216.summarize_directional_moves(enriched)
    overlap_summary, trade_overlap = p216.summarize_trade_overlap(enriched)
    churn_summary, churn_chains = p216.summarize_churn(enriched, gap_minutes=float(args.reentry_gap_minutes))
    hold_summary, hold_counterfactuals = p216.summarize_churn_hold_counterfactuals(enriched, churn_chains)
    top_differences = p216.summarize_day_differences(enriched)

    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "diagnostic / attribution audit",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_PREMIUM_LEANING_BLENDED_UTILITY_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "other_baseline_label": "none",
        "data_used": {
            "challenger_trades": str(args.challenger_trades),
            "protocol101_trades": str(args.protocol101_trades),
            "recent_protocol101_trades": str(args.recent_protocol101_trades),
            "normalized_dir": str(args.normalized_dir),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "reentry_gap_minutes": float(args.reentry_gap_minutes),
        "row_counts": {
            "challenger_trades": int(len(challenger)),
            "protocol101_trades": int(len(protocol101)),
            "enriched_trades": int(len(enriched)),
            "quote_skip_rows": int(len(quote_skips)),
            "trade_overlap_rows": int(len(trade_overlap)),
            "churn_chains": int(len(churn_chains)),
            "hold_counterfactual_rows": int(len(hold_counterfactuals)),
        },
        "headline": p216.headline(enriched),
        "side_summary": side_summary,
        "time_bucket_summary": time_summary,
        "exit_reason_summary": exit_summary,
        "directional_move_summary": move_summary,
        "trade_overlap_summary": overlap_summary,
        "churn_summary": churn_summary,
        "churn_hold_counterfactual_summary": hold_summary,
        "top_day_differences": top_differences,
        "quote_skip_counts": p216.count_by(quote_skips, "skip_reason"),
        "decision": decide(enriched, churn_summary, hold_summary),
        "next_experiment": (
            "Use the runtime parity pass plus this attribution to draft a replacement-readiness checklist. "
            "Do not replace PAPER_DEFAULT_PROTOCOL101 until live/no-order feature parity is demonstrated."
        ),
    }

    p216.write_csv(args.out_dir / "side_summary.csv", side_summary)
    p216.write_csv(args.out_dir / "time_bucket_summary.csv", time_summary)
    p216.write_csv(args.out_dir / "exit_reason_summary.csv", exit_summary)
    p216.write_csv(args.out_dir / "directional_move_summary.csv", move_summary)
    p216.write_csv(args.out_dir / "trade_overlap_summary.csv", overlap_summary)
    trade_overlap.to_csv(args.out_dir / "trade_overlap_rows.csv", index=False)
    p216.write_csv(args.out_dir / "churn_summary.csv", churn_summary)
    churn_chains.to_csv(args.out_dir / "churn_chains.csv", index=False)
    p216.write_csv(args.out_dir / "churn_hold_counterfactual_summary.csv", hold_summary)
    hold_counterfactuals.to_csv(args.out_dir / "churn_hold_counterfactuals.csv", index=False)
    p216.write_csv(args.out_dir / "top_day_differences.csv", top_differences)
    enriched.to_csv(args.out_dir / "enriched_policy_trades.csv", index=False)
    pd.DataFrame(quote_skips).to_csv(args.out_dir / "quote_skips.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    p216.write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def decide(enriched: pd.DataFrame, churn_summary: list[dict[str, object]], hold_summary: list[dict[str, object]]) -> str:
    base = p216.decide(enriched, churn_summary, hold_summary)
    if base == "attribution_supports_challenger_next_runtime_parity":
        return "attribution_supports_challenger_paper_default_unchanged"
    return base


if __name__ == "__main__":
    raise SystemExit(main())
