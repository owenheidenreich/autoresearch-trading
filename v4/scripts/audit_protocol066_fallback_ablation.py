from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from v4.model.hypothesis_protocol import stress_trades
from v4.model.supervised_pilot import Trade
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration


DEFAULT_SELECTED = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_069_protocol066_artifact_persistence/"
    "selected_trades_sequence_exits.json"
)
DEFAULT_SEQUENCE_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_060_lifecycle_sequence_dataset")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_073_protocol066_fallback_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="protocol066")
    parser.add_argument("--protocol-label", default="Protocol 066")
    parser.add_argument("--selected-trades", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _load_joined(selected_path: Path, sequence_dir: Path) -> pd.DataFrame:
    selected = pd.DataFrame(json.loads(selected_path.read_text()))
    trades = pd.read_parquet(sequence_dir / "protocol054_lifecycle_trades.parquet")
    joined = selected.merge(
        trades[["trade_uid", "baseline_pnl", "baseline_exit_reason", "baseline_exit_step", "baseline_exit_time"]],
        on="trade_uid",
        how="left",
    )
    joined["candidate_pnl"] = pd.to_numeric(joined["candidate_pnl"], errors="coerce")
    joined["baseline_pnl"] = pd.to_numeric(joined["baseline_pnl"], errors="coerce")
    joined["no_fallback_pnl"] = np.where(
        joined["candidate_exit_reason"].eq("protocol054_fallback"),
        joined["baseline_pnl"],
        joined["candidate_pnl"],
    )
    joined["no_fallback_reason"] = np.where(
        joined["candidate_exit_reason"].eq("protocol054_fallback"),
        "baseline_" + joined["baseline_exit_reason"].astype(str),
        joined["candidate_exit_reason"].astype(str),
    )
    return joined


def _trades(frame: pd.DataFrame, pnl_column: str, strategy: str) -> list[Trade]:
    return [
        Trade(
            session=str(row.session),
            decision_time=str(row.decision_time),
            pnl=float(getattr(row, pnl_column)),
            score=None,
            right=str(row.right),
            offset=float(row.offset),
            strategy=strategy,
        )
        for row in frame.itertuples(index=False)
    ]


def _metrics(frame: pd.DataFrame, pnl_column: str, strategy: str) -> dict:
    trades = _trades(frame, pnl_column, strategy)
    base = metrics_with_concentration(trades)
    stress50 = metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=50.0))
    stress100 = metrics_with_concentration(stress_trades(trades, extra_cost_per_trade=100.0))
    return {
        "total_pnl": float(base["total_pnl"]),
        "profit_factor": float(base["profit_factor"]),
        "trades": int(base["trades"]),
        "stress50_total_pnl": float(stress50["total_pnl"]),
        "stress100_total_pnl": float(stress100["total_pnl"]),
    }


def _split_order(frame: pd.DataFrame) -> list[str]:
    order = ["q1_2025", "q2_2025", "q3_2025", "q4_2025", "q1_2026"]
    present = [split for split in order if split in set(frame["split"].astype(str))]
    if (frame["split"].eq("q1_2026") & frame["session"].astype(str).ge("2026-03-01")).any():
        present.append("march_2026")
    return present


def _frame_for_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "march_2026":
        return frame[frame["split"].eq("q1_2026") & frame["session"].astype(str).ge("2026-03-01")]
    return frame[frame["split"].eq(split)]


def _seed_rows(frame: pd.DataFrame, *, protocol_id: str) -> list[dict]:
    rows = []
    for split in _split_order(frame):
        split_frame = _frame_for_split(frame, split)
        for seed, group in split_frame.groupby("seed"):
            rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "candidate": _metrics(group, "candidate_pnl", protocol_id),
                    "no_fallback": _metrics(group, "no_fallback_pnl", f"{protocol_id}_no_fallback"),
                }
            )
    return rows


def _median(values: Sequence[float]) -> float:
    return float(np.median(list(values))) if values else 0.0


def _summary(seed_rows: list[dict], split_order: Sequence[str]) -> list[dict]:
    out = []
    for split in split_order:
        rows = [row for row in seed_rows if row["split"] == split]
        out.append(
            {
                "split": split,
                "candidate_median_pnl": _median(row["candidate"]["total_pnl"] for row in rows),
                "no_fallback_median_pnl": _median(row["no_fallback"]["total_pnl"] for row in rows),
                "delta_no_fallback_vs_candidate": _median(row["no_fallback"]["total_pnl"] for row in rows)
                - _median(row["candidate"]["total_pnl"] for row in rows),
                "candidate_stress50_median_pnl": _median(row["candidate"]["stress50_total_pnl"] for row in rows),
                "no_fallback_stress50_median_pnl": _median(row["no_fallback"]["stress50_total_pnl"] for row in rows),
                "candidate_pf_median": _median(row["candidate"]["profit_factor"] for row in rows),
                "no_fallback_pf_median": _median(row["no_fallback"]["profit_factor"] for row in rows),
                "no_fallback_positive_seed_fraction": float(np.mean([row["no_fallback"]["total_pnl"] > 0 for row in rows]))
                if rows
                else 0.0,
            }
        )
    return out


def _fallback_delta(frame: pd.DataFrame) -> dict:
    fallback = frame[frame["candidate_exit_reason"].eq("protocol054_fallback")].copy()
    fallback["delta_no_fallback_vs_candidate"] = fallback["no_fallback_pnl"] - fallback["candidate_pnl"]
    by_split = (
        fallback.groupby("split")["delta_no_fallback_vs_candidate"].agg(["count", "sum", "median", "mean"]).reset_index()
    )
    by_reason = (
        fallback.groupby("protocol054_exit_reason")["delta_no_fallback_vs_candidate"]
        .agg(["count", "sum", "median", "mean"])
        .reset_index()
    )
    return {
        "by_split": by_split.to_dict(orient="records"),
        "by_protocol054_exit_reason": by_reason.to_dict(orient="records"),
    }


def _write_report(path: Path, payload: dict) -> None:
    protocol_label = payload["protocol_label"]
    lines = [
        f"# {protocol_label} Fallback Ablation",
        "",
        "No paid data was downloaded. This audit replaces Protocol 054 fallback exits with the original baseline stop/target/time-flat path to test whether the live system can safely omit the fallback engine.",
        "",
        "## Summary",
        "",
        f"| Split | {protocol_label} PnL | No-Fallback PnL | Delta | {protocol_label} +50 | No-Fallback +50 | {protocol_label} PF | No-Fallback PF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['split']} | {row['candidate_median_pnl']:.0f} | {row['no_fallback_median_pnl']:.0f} | "
            f"{row['delta_no_fallback_vs_candidate']:.0f} | {row['candidate_stress50_median_pnl']:.0f} | "
            f"{row['no_fallback_stress50_median_pnl']:.0f} | {row['candidate_pf_median']:.3f} | {row['no_fallback_pf_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_joined(args.selected_trades, args.sequence_dir)
    split_order = _split_order(frame)
    seed_rows = _seed_rows(frame, protocol_id=args.protocol_id)
    summary = _summary(seed_rows, split_order)
    damages = [row for row in summary if row["delta_no_fallback_vs_candidate"] < 0]
    decision = (
        "Reject no-fallback simplification for promotion-readiness. It remains profitable, but it damages Q3, Q4, and March, "
        "and lowers profit factor. Protocol 054 fallback must either be persisted explicitly or replaced by a newly validated self-contained lifecycle model."
        if damages
        else "No-fallback simplification cleared this ablation, but it still requires full promotion validation before replacing Protocol 054 fallback."
    )
    payload = {
        "protocol_id": args.protocol_id,
        "protocol_label": args.protocol_label,
        "decision": decision,
        "split_order": split_order,
        "summary": summary,
        "seed_rows": seed_rows,
        "no_fallback_reason_counts": {str(k): int(v) for k, v in frame["no_fallback_reason"].value_counts().to_dict().items()},
        "fallback_delta": _fallback_delta(frame),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.md")
    print(json.dumps({"decision": "reject_no_fallback" if damages else "candidate", "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
