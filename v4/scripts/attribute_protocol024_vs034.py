"""Attribution between Protocol 024 and Protocol 034.

This is a no-paid-data diagnostic. It explains what the Protocol 034
contract-quality gate changed relative to the frozen Protocol 024 selected
trades, with March 2026 called out explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


FEATURES = (
    "quality_score",
    "threshold",
    "feature_contract_value_score",
    "feature_theta_burden_hold",
    "feature_spread_tax",
    "feature_breakeven_atr",
    "feature_gamma_theta_ratio_scaled",
    "feature_convexity_per_premium",
    "feature_abs_delta",
    "feature_pattern_count_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol024-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_024_side_value_multitask_nofee/selected_trades_enriched"),
    )
    parser.add_argument(
        "--protocol034-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_034_trade_preserving_quality_gate"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_protocol_024_vs_034_attribution"),
    )
    return parser.parse_args()


def _load_protocol024(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("selected_trades_*.json")):
        data = json.loads(path.read_text())
        for row in data:
            rows.append({**row, "protocol": "Protocol 024", "source_file": str(path)})
    if not rows:
        raise SystemExit(f"no Protocol 024 selected_trades_*.json under {directory}")
    return rows


def _proposal_key(row: dict) -> tuple:
    return (
        int(row["seed"]),
        str(row["split"]),
        str(row["session"]),
        str(row["decision_time"]),
        str(row["right"]),
        round(float(row["offset"]), 4),
        round(float(row["pnl"]), 4),
    )


def _exact_key(row: dict) -> tuple:
    return (
        int(row["seed"]),
        str(row["split"]),
        str(row["session"]),
        str(row["decision_time"]),
        str(row.get("contract_id", "")),
    )


def _minute_key(row: dict) -> tuple:
    return (
        int(row["seed"]),
        str(row["split"]),
        str(row["session"]),
        str(row["decision_time"]),
    )


def _load_proposals(directory: Path) -> list[dict]:
    rows: list[dict] = []
    proposal_dir = directory / "proposal_attribution"
    for path in sorted(proposal_dir.glob("seed*_*.json")):
        data = json.loads(path.read_text())
        for row in data:
            rows.append({**row, "source_file": str(path)})
    if not rows:
        raise SystemExit(f"no Protocol 034 proposal attribution files under {proposal_dir}")
    return rows


def _load_protocol034(directory: Path, proposal_by_key: dict[tuple, dict]) -> list[dict]:
    rows: list[dict] = []
    selected_dir = directory / "selected_trades"
    for path in sorted(selected_dir.glob("seed*_*.json")):
        data = json.loads(path.read_text())
        for row in data:
            proposal = proposal_by_key.get(_proposal_key(row), {})
            merged = {
                **proposal,
                **row,
                "contract_id": proposal.get("contract_id", ""),
                "protocol": "Protocol 034",
                "source_file": str(path),
            }
            rows.append(merged)
    if not rows:
        raise SystemExit(f"no Protocol 034 selected trade files under {selected_dir}")
    return rows


def _finite(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sum_pnl(rows: Sequence[dict]) -> float:
    return float(sum(float(row["pnl"]) for row in rows))


def _metrics(rows: Sequence[dict]) -> dict:
    pnl = np.asarray([float(row["pnl"]) for row in rows], dtype=float)
    if len(pnl) == 0:
        return {"trades": 0, "pnl": 0.0, "avg_pnl": 0.0, "median_pnl": 0.0, "win_rate": 0.0}
    return {
        "trades": int(len(pnl)),
        "pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "win_rate": float(np.mean(pnl > 0.0)),
    }


def _seed_medians(rows: Sequence[dict]) -> dict:
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if not by_seed:
        return {"seed_pnl_median": 0.0, "seed_trades_median": 0.0, "positive_seed_fraction": 0.0}
    pnls = [_sum_pnl(seed_rows) for seed_rows in by_seed.values()]
    trades = [len(seed_rows) for seed_rows in by_seed.values()]
    return {
        "seed_pnl_median": float(np.median(pnls)),
        "seed_trades_median": float(np.median(trades)),
        "positive_seed_fraction": float(np.mean([pnl > 0.0 for pnl in pnls])),
    }


def _feature_summary(rows: Sequence[dict]) -> dict:
    out = {}
    for feature in FEATURES:
        values = [_finite(row.get(feature)) for row in rows]
        clean = np.asarray([value for value in values if value is not None], dtype=float)
        out[f"{feature}_median"] = float(np.median(clean)) if len(clean) else None
        out[f"{feature}_mean"] = float(np.mean(clean)) if len(clean) else None
    return out


def _side_bucket(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        bucket = str(row.get("bucket") or _bucket_from_time(str(row["decision_time"])))
        groups.setdefault((bucket, str(row.get("right"))), []).append(row)
    return [
        {"bucket": bucket, "right": right, **_metrics(group)}
        for (bucket, right), group in sorted(groups.items())
    ]


def _bucket_from_time(value: str) -> str:
    # Decision times are UTC ISO strings. This fallback is intentionally simple:
    # selected Protocol 034 files do not store bucket, but proposal rows do.
    return "unknown"


def _reason_for_baseline_only(row: dict) -> str:
    if row.get("proposal_missing"):
        return "proposal_missing"
    if not bool(row.get("passes_quality_threshold", False)):
        return "quality_rejected"
    return "cooldown_or_trade_limit_after_gate"


def _comparison_for_split(
    *,
    split: str,
    baseline_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    proposal_by_exact: dict[tuple, dict],
) -> dict:
    baseline = [row for row in baseline_rows if row["split"] == split]
    candidate = [row for row in candidate_rows if row["split"] == split]
    candidate_by_exact = {_exact_key(row): row for row in candidate}
    baseline_by_exact = {_exact_key(row): row for row in baseline}
    exact_keys = set(baseline_by_exact) & set(candidate_by_exact)
    exact_overlap = [baseline_by_exact[key] for key in exact_keys]

    candidate_minutes = {_minute_key(row) for row in candidate}
    baseline_minutes = {_minute_key(row) for row in baseline}
    baseline_only = []
    same_minute_changed = []
    for row in baseline:
        key = _exact_key(row)
        minute = _minute_key(row)
        if key in exact_keys:
            continue
        enriched = {**row}
        proposal = proposal_by_exact.get(key)
        if proposal is None:
            enriched["proposal_missing"] = True
        else:
            enriched.update({k: v for k, v in proposal.items() if k not in enriched or k in {"quality_score", "threshold", "passes_quality_threshold"}})
        if minute in candidate_minutes:
            same_minute_changed.append(enriched)
        else:
            baseline_only.append(enriched)
    candidate_only = [
        row for row in candidate if _exact_key(row) not in exact_keys and _minute_key(row) not in baseline_minutes
    ]
    changed_candidate = [
        row for row in candidate if _exact_key(row) not in exact_keys and _minute_key(row) in baseline_minutes
    ]
    for row in baseline_only:
        row["attribution_reason"] = _reason_for_baseline_only(row)
    missed_winners = [row for row in baseline_only if float(row["pnl"]) > 0.0]
    avoided_losers = [row for row in baseline_only if float(row["pnl"]) < 0.0]
    added_winners = [row for row in candidate_only if float(row["pnl"]) > 0.0]
    added_losers = [row for row in candidate_only if float(row["pnl"]) < 0.0]
    reason_rows = []
    for reason in sorted({row["attribution_reason"] for row in baseline_only}):
        group = [row for row in baseline_only if row["attribution_reason"] == reason]
        reason_rows.append({"reason": reason, **_metrics(group), **_feature_summary(group)})
    return {
        "split": split,
        "baseline": _metrics(baseline) | _seed_medians(baseline),
        "candidate": _metrics(candidate) | _seed_medians(candidate),
        "pnl_delta": _sum_pnl(candidate) - _sum_pnl(baseline),
        "exact_overlap": _metrics(exact_overlap),
        "same_minute_changed_baseline": _metrics(same_minute_changed),
        "same_minute_changed_candidate": _metrics(changed_candidate),
        "baseline_only": _metrics(baseline_only) | _feature_summary(baseline_only),
        "candidate_only": _metrics(candidate_only) | _feature_summary(candidate_only),
        "missed_winners": _metrics(missed_winners) | _feature_summary(missed_winners),
        "avoided_losers": _metrics(avoided_losers) | _feature_summary(avoided_losers),
        "added_winners": _metrics(added_winners) | _feature_summary(added_winners),
        "added_losers": _metrics(added_losers) | _feature_summary(added_losers),
        "reason_summary": reason_rows,
        "baseline_side_bucket": _side_bucket(baseline),
        "candidate_side_bucket": _side_bucket(candidate),
        "baseline_only_side_bucket": _side_bucket(baseline_only),
        "candidate_only_side_bucket": _side_bucket(candidate_only),
        "missed_winner_examples": sorted(missed_winners, key=lambda row: float(row["pnl"]), reverse=True)[:20],
        "avoided_loser_examples": sorted(avoided_losers, key=lambda row: float(row["pnl"]))[:20],
        "candidate_only_examples": sorted(candidate_only, key=lambda row: abs(float(row["pnl"])), reverse=True)[:20],
    }


def _fmt(value: object, digits: int = 0) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.{digits}f}"


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 024 vs 034 Attribution",
        "",
        payload["framing"],
        "",
        "## Split Decomposition",
        "",
        "| Split | 024 PnL | 034 PnL | Delta | 024 Trades | 034 Trades | Exact Overlap | Missed Winners | Avoided Losers | Added Winners | Added Losers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["splits"]:
        lines.append(
            f"| {row['split']} | {_fmt(row['baseline']['pnl'])} | {_fmt(row['candidate']['pnl'])} | {_fmt(row['pnl_delta'])} | "
            f"{_fmt(row['baseline']['trades'])} | {_fmt(row['candidate']['trades'])} | {_fmt(row['exact_overlap']['trades'])} | "
            f"{_fmt(row['missed_winners']['pnl'])}/{_fmt(row['missed_winners']['trades'])} | "
            f"{_fmt(row['avoided_losers']['pnl'])}/{_fmt(row['avoided_losers']['trades'])} | "
            f"{_fmt(row['added_winners']['pnl'])}/{_fmt(row['added_winners']['trades'])} | "
            f"{_fmt(row['added_losers']['pnl'])}/{_fmt(row['added_losers']['trades'])} |"
        )
    march = payload["march"]
    lines += [
        "",
        "## March Focus",
        "",
        f"- All-seed March delta: `{_fmt(march['pnl_delta'])}`.",
        f"- Protocol 034 removed `{_fmt(march['baseline_only']['trades'])}` Protocol 024 trades worth `{_fmt(march['baseline_only']['pnl'])}`.",
        f"- Removed winners: `{_fmt(march['missed_winners']['trades'])}` trades worth `{_fmt(march['missed_winners']['pnl'])}`.",
        f"- Avoided losers: `{_fmt(march['avoided_losers']['trades'])}` trades worth `{_fmt(march['avoided_losers']['pnl'])}`.",
        f"- Added trades after cooldown shifts: `{_fmt(march['candidate_only']['trades'])}` trades worth `{_fmt(march['candidate_only']['pnl'])}`.",
        "",
        "## March Removed Trade Reasons",
        "",
        "| Reason | Trades | PnL | Win Rate | Median Quality | Median Value | Median Theta | Median Spread | Median Breakeven | Median Gamma/Theta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in march["reason_summary"]:
        lines.append(
            f"| {row['reason']} | {_fmt(row['trades'])} | {_fmt(row['pnl'])} | {_fmt(row['win_rate'], 2)} | "
            f"{_fmt(row['quality_score_median'], 3)} | {_fmt(row['feature_contract_value_score_median'], 3)} | "
            f"{_fmt(row['feature_theta_burden_hold_median'], 3)} | {_fmt(row['feature_spread_tax_median'], 3)} | "
            f"{_fmt(row['feature_breakeven_atr_median'], 2)} | {_fmt(row['feature_gamma_theta_ratio_scaled_median'], 3)} |"
        )
    lines += [
        "",
        "## March Side / Time Exposure",
        "",
        "Protocol 024:",
        "",
        "| Bucket | Side | Trades | PnL | Win Rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in march["baseline_side_bucket"]:
        lines.append(f"| {row['bucket']} | {row['right']} | {_fmt(row['trades'])} | {_fmt(row['pnl'])} | {_fmt(row['win_rate'], 2)} |")
    lines += [
        "",
        "Protocol 034:",
        "",
        "| Bucket | Side | Trades | PnL | Win Rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in march["candidate_side_bucket"]:
        lines.append(f"| {row['bucket']} | {row['right']} | {_fmt(row['trades'])} | {_fmt(row['pnl'])} | {_fmt(row['win_rate'], 2)} |")
    lines += [
        "",
        "## March Missed Winners",
        "",
        "| Seed | Session | Time | Side | Offset | PnL | Quality | Threshold | Value | Theta | Spread | Breakeven | Gamma/Theta | Pattern | Reason |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in march["missed_winner_examples"][:12]:
        lines.append(
            f"| {row['seed']} | {row['session']} | {row['decision_time']} | {row['right']} | {_fmt(row['offset'])} | {_fmt(row['pnl'])} | "
            f"{_fmt(row.get('quality_score'), 3)} | {_fmt(row.get('threshold'), 2)} | {_fmt(row.get('feature_contract_value_score'), 3)} | "
            f"{_fmt(row.get('feature_theta_burden_hold'), 3)} | {_fmt(row.get('feature_spread_tax'), 3)} | "
            f"{_fmt(row.get('feature_breakeven_atr'), 2)} | {_fmt(row.get('feature_gamma_theta_ratio_scaled'), 3)} | "
            f"{_fmt(row.get('feature_pattern_count_norm'), 3)} | {row.get('attribution_reason')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def _sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "inf" if value > 0 else "-inf"
    return value


def main() -> int:
    args = parse_args()
    protocol024 = _load_protocol024(args.protocol024_dir)
    proposals = _load_proposals(args.protocol034_dir)
    proposal_by_key = {_proposal_key(row): row for row in proposals}
    proposal_by_exact = {_exact_key(row): row for row in proposals}
    protocol034 = _load_protocol034(args.protocol034_dir, proposal_by_key)
    split_order = ["selection", "march_2026", "q1_2025", "q2_2025", "q3_2025", "q4_2025"]
    splits = [
        _comparison_for_split(
            split=split,
            baseline_rows=protocol024,
            candidate_rows=protocol034,
            proposal_by_exact=proposal_by_exact,
        )
        for split in split_order
    ]
    march = next(row for row in splits if row["split"] == "march_2026")
    missed = march["missed_winners"]
    avoided = march["avoided_losers"]
    interpretation = (
        "Protocol 034's March failure is mainly a false-negative contract-quality problem, not a side flip. "
        "The gate removed profitable Protocol 024 entries that the quality model scored below threshold or displaced through cooldown shifts. "
        "Several missed winners had ordinary-looking A+ economics rather than obvious overpay flags, which means a static or proposal-level quality gate can reject convex winners that need room to express. "
        "The next model change should preserve Protocol 024's March trade availability and use contract quality as a soft representation/risk feature, not a hard entry veto, unless a March-safe rejection rule is proven on selection first."
    )
    payload = {
        "framing": (
            "No-paid-data attribution between Protocol 024 and Protocol 034. "
            "The goal is to explain why the trade-preserving quality gate improved most 2025 blocks but damaged March 2026."
        ),
        "splits": splits,
        "march": march,
        "march_missed_minus_avoided_pnl": float(missed["pnl"] + avoided["pnl"]),
        "interpretation": interpretation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = _sanitize(payload)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_report(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.json")
    print(args.out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
