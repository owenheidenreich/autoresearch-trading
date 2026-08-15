"""Compare two selected-trade sets trade by trade.

This is a no-paid-data attribution tool. It compares a baseline protocol and a
candidate protocol on exact trade overlap, same-minute contract changes, avoided
losers, missed winners, added winners, and added losers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


_FEATURES = (
    "feature_contract_value_score",
    "feature_theta_burden_hold",
    "feature_spread_tax",
    "feature_breakeven_atr",
    "feature_gamma_theta_ratio_scaled",
    "feature_abs_delta",
    "feature_pattern_count_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_021_q1_fragility_diagnostic/selected_trades_enriched"),
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_024_side_value_multitask/selected_trades_enriched"),
    )
    parser.add_argument("--baseline-name", default="Protocol 018")
    parser.add_argument("--candidate-name", default="Protocol 024")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_protocol_018_vs_024_trade_comparison"),
    )
    return parser.parse_args()


def _load_dir(path: Path, *, protocol: str) -> list[dict]:
    rows: list[dict] = []
    files = sorted(path.glob("selected_trades_*.json"))
    if not files:
        files = sorted(path.glob("*.json"))
    for file_path in files:
        data = json.loads(file_path.read_text())
        for row in data:
            rows.append({**row, "protocol": protocol, "source_file": str(file_path)})
    if not rows:
        raise SystemExit(f"no selected trade JSON files found under {path}")
    return rows


def _exact_key(row: dict) -> tuple:
    return (
        int(row["seed"]),
        str(row["split"]),
        str(row["session"]),
        str(row["decision_time"]),
        str(row["contract_id"]),
    )


def _minute_key(row: dict) -> tuple:
    return (
        int(row["seed"]),
        str(row["split"]),
        str(row["session"]),
        str(row["decision_time"]),
    )


def _finite(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _median_feature(rows: Sequence[dict], feature: str) -> float | None:
    values = [_finite(row.get(feature)) for row in rows]
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    if len(clean) == 0:
        return None
    return float(np.median(clean))


def _feature_summary(rows: Sequence[dict]) -> dict:
    out = {}
    for feature in _FEATURES:
        out[f"{feature}_median"] = _median_feature(rows, feature)
    return out


def _pnl_sum(rows: Sequence[dict]) -> float:
    return float(sum(float(row["pnl"]) for row in rows))


def _win_rate(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row["pnl"]) > 0.0 for row in rows]))


def _metrics(rows: Sequence[dict]) -> dict:
    pnl = np.asarray([float(row["pnl"]) for row in rows], dtype=float)
    if len(pnl) == 0:
        return {"trades": 0, "pnl": 0.0, "avg_pnl": 0.0, "median_pnl": 0.0, "win_rate": 0.0}
    return {
        "trades": int(len(rows)),
        "pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "win_rate": _win_rate(rows),
    }


def _seed_medians(rows: Sequence[dict]) -> dict:
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    if not by_seed:
        return {"seed_pnl_median": 0.0, "seed_trades_median": 0.0, "positive_seed_fraction": 0.0}
    pnls = [_pnl_sum(seed_rows) for seed_rows in by_seed.values()]
    trades = [len(seed_rows) for seed_rows in by_seed.values()]
    return {
        "seed_pnl_median": float(np.median(pnls)),
        "seed_trades_median": float(np.median(trades)),
        "positive_seed_fraction": float(np.mean([pnl > 0.0 for pnl in pnls])),
    }


def _side_bucket_summary(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row.get("bucket")), str(row.get("right"))), []).append(row)
    out = []
    for (bucket, right), group in sorted(groups.items()):
        out.append({"bucket": bucket, "right": right, **_metrics(group)})
    return out


def _summarize_category(rows: Sequence[dict], prefix: str) -> dict:
    out = {f"{prefix}_{key}": value for key, value in _metrics(rows).items()}
    for key, value in _feature_summary(rows).items():
        out[f"{prefix}_{key}"] = value
    return out


def _build_comparison(baseline_rows: Sequence[dict], candidate_rows: Sequence[dict]) -> dict:
    baseline_by_exact = {_exact_key(row): row for row in baseline_rows}
    candidate_by_exact = {_exact_key(row): row for row in candidate_rows}
    baseline_by_minute = {_minute_key(row): row for row in baseline_rows}
    candidate_by_minute = {_minute_key(row): row for row in candidate_rows}

    exact_keys = set(baseline_by_exact) & set(candidate_by_exact)
    exact_baseline = [baseline_by_exact[key] for key in exact_keys]
    exact_candidate = [candidate_by_exact[key] for key in exact_keys]

    same_minute_keys = (set(baseline_by_minute) & set(candidate_by_minute)) - {
        _minute_key(row) for row in exact_baseline
    }
    same_minute_pairs = [
        {
            "baseline": baseline_by_minute[key],
            "candidate": candidate_by_minute[key],
            "pnl_delta": float(candidate_by_minute[key]["pnl"]) - float(baseline_by_minute[key]["pnl"]),
            "same_side": str(candidate_by_minute[key]["right"]) == str(baseline_by_minute[key]["right"]),
            "same_contract": str(candidate_by_minute[key]["contract_id"]) == str(baseline_by_minute[key]["contract_id"]),
        }
        for key in sorted(same_minute_keys)
    ]

    exact_minute_keys = {_minute_key(row) for row in exact_baseline}
    replaced_minute_keys = set(same_minute_keys)
    baseline_only = [
        row
        for key, row in baseline_by_minute.items()
        if key not in exact_minute_keys and key not in replaced_minute_keys
    ]
    candidate_only = [
        row
        for key, row in candidate_by_minute.items()
        if key not in exact_minute_keys and key not in replaced_minute_keys
    ]
    avoided_losers = [row for row in baseline_only if float(row["pnl"]) < 0.0]
    missed_winners = [row for row in baseline_only if float(row["pnl"]) > 0.0]
    added_winners = [row for row in candidate_only if float(row["pnl"]) > 0.0]
    added_losers = [row for row in candidate_only if float(row["pnl"]) < 0.0]

    same_minute_baseline_pnl = float(sum(float(pair["baseline"]["pnl"]) for pair in same_minute_pairs))
    same_minute_candidate_pnl = float(sum(float(pair["candidate"]["pnl"]) for pair in same_minute_pairs))
    baseline_pnl = _pnl_sum(baseline_rows)
    candidate_pnl = _pnl_sum(candidate_rows)
    explained_delta = (
        (_pnl_sum(exact_candidate) - _pnl_sum(exact_baseline))
        + (same_minute_candidate_pnl - same_minute_baseline_pnl)
        + _pnl_sum(candidate_only)
        - _pnl_sum(baseline_only)
    )
    return {
        "baseline": _metrics(baseline_rows) | _seed_medians(baseline_rows),
        "candidate": _metrics(candidate_rows) | _seed_medians(candidate_rows),
        "pnl_delta": float(candidate_pnl - baseline_pnl),
        "explained_delta": float(explained_delta),
        "exact_overlap": _metrics(exact_baseline),
        "same_minute_changed": {
            "trades": int(len(same_minute_pairs)),
            "baseline_pnl": same_minute_baseline_pnl,
            "candidate_pnl": same_minute_candidate_pnl,
            "pnl_delta": same_minute_candidate_pnl - same_minute_baseline_pnl,
            "same_side_fraction": float(np.mean([pair["same_side"] for pair in same_minute_pairs])) if same_minute_pairs else 0.0,
        },
        "baseline_only": _summarize_category(baseline_only, "baseline_only"),
        "candidate_only": _summarize_category(candidate_only, "candidate_only"),
        "avoided_losers": _summarize_category(avoided_losers, "avoided_losers"),
        "missed_winners": _summarize_category(missed_winners, "missed_winners"),
        "added_winners": _summarize_category(added_winners, "added_winners"),
        "added_losers": _summarize_category(added_losers, "added_losers"),
        "baseline_only_side_bucket": _side_bucket_summary(baseline_only),
        "candidate_only_side_bucket": _side_bucket_summary(candidate_only),
        "same_minute_pairs": same_minute_pairs,
        "baseline_only_examples": sorted(baseline_only, key=lambda row: float(row["pnl"]))[:20]
        + sorted(baseline_only, key=lambda row: float(row["pnl"]), reverse=True)[:20],
        "candidate_only_examples": sorted(candidate_only, key=lambda row: float(row["pnl"]))[:20]
        + sorted(candidate_only, key=lambda row: float(row["pnl"]), reverse=True)[:20],
    }


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


def _write_markdown(path: Path, payload: dict) -> None:
    baseline_name = payload["baseline_name"]
    candidate_name = payload["candidate_name"]
    lines = [
        f"# {baseline_name} vs {candidate_name} Trade Comparison",
        "",
        "No paid data. Compares selected trades by seed, split, timestamp, and contract.",
        "",
        "## Split Summary",
        "",
        f"| Split | {baseline_name} PnL | {candidate_name} PnL | Delta | Exact Overlap | Same-Minute Changed | Baseline-Only PnL | Candidate-Only PnL | Avoided Losers | Missed Winners |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["by_split"].items():
        lines.append(
            f"| {split} | {_fmt(row['baseline']['pnl'])} | {_fmt(row['candidate']['pnl'])} | "
            f"{_fmt(row['pnl_delta'])} | {_fmt(row['exact_overlap']['trades'])} | "
            f"{_fmt(row['same_minute_changed']['trades'])} | {_fmt(row['baseline_only']['baseline_only_pnl'])} | "
            f"{_fmt(row['candidate_only']['candidate_only_pnl'])} | "
            f"{_fmt(abs(row['avoided_losers']['avoided_losers_pnl']))} | "
            f"{_fmt(row['missed_winners']['missed_winners_pnl'])} |"
        )
    lines += [
        "",
        "## Attribution",
        "",
        "| Split | Same-Minute Delta | Added Winners | Added Losers | Avoided Losers | Missed Winners | Net Only-Trade Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["by_split"].items():
        net_only = row["candidate_only"]["candidate_only_pnl"] - row["baseline_only"]["baseline_only_pnl"]
        lines.append(
            f"| {split} | {_fmt(row['same_minute_changed']['pnl_delta'])} | "
            f"{_fmt(row['added_winners']['added_winners_pnl'])} | {_fmt(row['added_losers']['added_losers_pnl'])} | "
            f"{_fmt(abs(row['avoided_losers']['avoided_losers_pnl']))} | "
            f"{_fmt(row['missed_winners']['missed_winners_pnl'])} | {_fmt(net_only)} |"
        )
    lines += [
        "",
        "## Side / Time Of Only Trades",
        "",
        f"### {baseline_name}-Only Trades",
        "",
        "| Split | Bucket | Side | Trades | PnL | Win Rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split, row in payload["by_split"].items():
        for item in row["baseline_only_side_bucket"]:
            lines.append(
                f"| {split} | {item['bucket']} | {item['right']} | {_fmt(item['trades'])} | "
                f"{_fmt(item['pnl'])} | {_fmt(item['win_rate'], 2)} |"
            )
    lines += [
        "",
        f"### {candidate_name}-Only Trades",
        "",
        "| Split | Bucket | Side | Trades | PnL | Win Rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split, row in payload["by_split"].items():
        for item in row["candidate_only_side_bucket"]:
            lines.append(
                f"| {split} | {item['bucket']} | {item['right']} | {_fmt(item['trades'])} | "
                f"{_fmt(item['pnl'])} | {_fmt(item['win_rate'], 2)} |"
            )
    lines += [
        "",
        "## Contract-Quality Attribution",
        "",
        "| Split | Category | Trades | PnL | Value Score | Theta Burden | Breakeven ATR | Gamma/Theta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    categories = (
        ("avoided_losers", "Avoided Losers"),
        ("missed_winners", "Missed Winners"),
        ("added_winners", "Added Winners"),
        ("added_losers", "Added Losers"),
    )
    for split, row in payload["by_split"].items():
        for key, label in categories:
            item = row[key]
            lines.append(
                f"| {split} | {label} | {_fmt(item[f'{key}_trades'])} | {_fmt(item[f'{key}_pnl'])} | "
                f"{_fmt(item.get(f'{key}_feature_contract_value_score_median'), 3)} | "
                f"{_fmt(item.get(f'{key}_feature_theta_burden_hold_median'), 3)} | "
                f"{_fmt(item.get(f'{key}_feature_breakeven_atr_median'), 2)} | "
                f"{_fmt(item.get(f'{key}_feature_gamma_theta_ratio_scaled_median'), 3)} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def _interpret(by_split: dict[str, dict]) -> str:
    improved = [split for split, row in by_split.items() if float(row.get("pnl_delta", 0.0)) > 0.0]
    weakened = [split for split, row in by_split.items() if float(row.get("pnl_delta", 0.0)) < 0.0]
    if improved and weakened:
        return (
            "Mixed trade-set attribution: the candidate improves "
            f"{', '.join(improved)} but weakens {', '.join(weakened)}. Treat this as diagnostic evidence, "
            "not an automatic promotion or rejection. The next step is to inspect which only-trade side/time "
            "buckets and contract-quality bands explain the split-level changes."
        )
    if improved:
        return (
            f"The candidate improves every compared split ({', '.join(improved)}). This supports moving to "
            "the next validation gate, subject to path-level and stress checks."
        )
    if weakened:
        return (
            f"The candidate weakens every compared split ({', '.join(weakened)}). Reject it for promotion unless "
            "a separate risk-control objective explains the tradeoff."
        )
    return "No material split-level PnL delta; keep this as a neutral attribution report."


def main() -> int:
    args = parse_args()
    baseline = _load_dir(args.baseline_dir, protocol=args.baseline_name)
    candidate = _load_dir(args.candidate_dir, protocol=args.candidate_name)
    splits = sorted(set(row["split"] for row in baseline) | set(row["split"] for row in candidate))
    by_split = {}
    for split in splits:
        base_rows = [row for row in baseline if row["split"] == split]
        cand_rows = [row for row in candidate if row["split"] == split]
        by_split[split] = _build_comparison(base_rows, cand_rows)
    payload = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "by_split": by_split,
        "interpretation": _interpret(by_split),
    }
    payload = _sanitize(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.json")
    print(args.out_dir / "report.md")
    print(payload["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
