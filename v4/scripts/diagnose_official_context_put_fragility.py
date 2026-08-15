"""Diagnose post-open put trade-set changes after official context rebuild.

This is a no-paid-data attribution tool. It compares the frozen proxy-context
Protocol 039 selected trades with the frozen official-context Protocol 039
selected trades and focuses on post-open morning puts, where Protocol 042 found
the largest Q3/Q4 degradation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FOCUS_BUCKET = "post_open_morning"
FOCUS_RIGHT = "P"

FEATURES = (
    "feature_contract_value_score",
    "feature_theta_burden_hold",
    "feature_premium_decay_burden",
    "feature_spread_tax",
    "feature_breakeven_atr",
    "feature_gamma_theta_ratio_scaled",
    "feature_convexity_per_premium",
    "feature_abs_delta",
    "feature_gamma_abs",
    "feature_theta_abs",
    "feature_iv",
    "feature_liquidity_score",
    "feature_log_option_volume",
    "feature_log_open_interest",
    "feature_pattern_count_norm",
    "feature_pattern_vwap_hold_continuation",
    "feature_pattern_vwap_pullback_resume",
    "feature_pattern_vwap_reclaim",
    "feature_pattern_omar_mid_reclaim",
    "feature_pattern_omar_retest_bounce",
    "feature_pattern_momentum_ignition",
    "feature_pattern_pullback_resume",
    "feature_pattern_sigma_trend_continuation",
    "feature_pattern_compression_breakout",
    "feature_pattern_compression_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_039_broader_baseline_validation/selected_trades"),
    )
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path(
            "v4/audit/autoresearch/v4_aplus_hypothesis_039_broader_baseline_validation_official_context/selected_trades"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_043_post_open_put_fragility_diagnostic"),
    )
    return parser.parse_args()


def _load_dir(path: Path, *, protocol: str) -> list[dict]:
    rows: list[dict] = []
    files = sorted(path.glob("*.json"))
    if not files:
        raise SystemExit(f"no selected trade JSON files found under {path}")
    for file_path in files:
        for row in json.loads(file_path.read_text()):
            rows.append({**row, "protocol": protocol, "source_file": str(file_path)})
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


def _only_rows(proxy_rows: Sequence[dict], official_rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    proxy_by_exact = {_exact_key(row): row for row in proxy_rows}
    official_by_exact = {_exact_key(row): row for row in official_rows}
    proxy_by_minute = {_minute_key(row): row for row in proxy_rows}
    official_by_minute = {_minute_key(row): row for row in official_rows}

    exact_keys = set(proxy_by_exact) & set(official_by_exact)
    exact_minute_keys = {_minute_key(proxy_by_exact[key]) for key in exact_keys}
    same_minute_keys = (set(proxy_by_minute) & set(official_by_minute)) - exact_minute_keys

    proxy_only = [
        row
        for key, row in proxy_by_minute.items()
        if key not in exact_minute_keys and key not in same_minute_keys
    ]
    official_only = [
        row
        for key, row in official_by_minute.items()
        if key not in exact_minute_keys and key not in same_minute_keys
    ]
    return proxy_only, official_only


def _focus(rows: Iterable[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if str(row.get("bucket")) == FOCUS_BUCKET and str(row.get("right")) == FOCUS_RIGHT
    ]


def _finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _values(rows: Sequence[dict], feature: str) -> np.ndarray:
    vals = [_finite(row.get(feature)) for row in rows]
    return np.asarray([val for val in vals if val is not None], dtype=float)


def _median(rows: Sequence[dict], feature: str) -> float | None:
    vals = _values(rows, feature)
    if len(vals) == 0:
        return None
    return float(np.median(vals))


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    signs = np.sign(a[:, None] - b[None, :])
    return float(signs.mean())


def _metric(rows: Sequence[dict]) -> dict:
    pnl = np.asarray([float(row["pnl"]) for row in rows], dtype=float)
    if len(pnl) == 0:
        return {"trades": 0, "pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0}
    return {
        "trades": int(len(rows)),
        "pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0.0).mean()),
        "avg_pnl": float(pnl.mean()),
    }


def _feature_table(winners: Sequence[dict], losers: Sequence[dict]) -> list[dict]:
    rows = []
    for feature in FEATURES:
        win_vals = _values(winners, feature)
        lose_vals = _values(losers, feature)
        if len(win_vals) == 0 or len(lose_vals) == 0:
            continue
        win_median = float(np.median(win_vals))
        lose_median = float(np.median(lose_vals))
        rows.append(
            {
                "feature": feature,
                "winner_median": win_median,
                "loser_median": lose_median,
                "median_diff_winner_minus_loser": float(win_median - lose_median),
                "cliffs_delta_winner_vs_loser": _cliffs_delta(win_vals, lose_vals),
            }
        )
    return rows


def _category_summary(rows: Sequence[dict]) -> dict:
    summary = _metric(rows)
    for feature in FEATURES:
        summary[f"{feature}_median"] = _median(rows, feature)
    return summary


def _build(proxy_rows: Sequence[dict], official_rows: Sequence[dict]) -> dict:
    splits = sorted(set(row["split"] for row in proxy_rows) | set(row["split"] for row in official_rows))
    by_split = {}
    feature_effects: dict[str, list[dict]] = {feature: [] for feature in FEATURES}
    for split in splits:
        proxy_split = [row for row in proxy_rows if row["split"] == split]
        official_split = [row for row in official_rows if row["split"] == split]
        proxy_only, official_only = _only_rows(proxy_split, official_split)
        proxy_focus = _focus(proxy_only)
        official_focus = _focus(official_only)
        missed_winners = [row for row in proxy_focus if float(row["pnl"]) > 0.0]
        avoided_losers = [row for row in proxy_focus if float(row["pnl"]) < 0.0]
        added_winners = [row for row in official_focus if float(row["pnl"]) > 0.0]
        added_losers = [row for row in official_focus if float(row["pnl"]) < 0.0]
        proxy_features = _feature_table(missed_winners, avoided_losers)
        official_features = _feature_table(added_winners, added_losers)
        for item in proxy_features:
            feature_effects[item["feature"]].append({"split": split, "source": "proxy_only", **item})
        by_split[split] = {
            "proxy_only_focus": _metric(proxy_focus),
            "official_only_focus": _metric(official_focus),
            "proxy_only_missed_winners": _category_summary(missed_winners),
            "proxy_only_avoided_losers": _category_summary(avoided_losers),
            "official_only_added_winners": _category_summary(added_winners),
            "official_only_added_losers": _category_summary(added_losers),
            "proxy_missed_vs_avoided_features": proxy_features,
            "official_added_winners_vs_losers_features": official_features,
        }
    stable = []
    for feature, effects in feature_effects.items():
        clean = [item for item in effects if item["cliffs_delta_winner_vs_loser"] is not None]
        signs = [int(np.sign(item["cliffs_delta_winner_vs_loser"])) for item in clean]
        nonzero = [sign for sign in signs if sign != 0]
        if not nonzero:
            continue
        majority_sign = 1 if sum(sign > 0 for sign in nonzero) >= sum(sign < 0 for sign in nonzero) else -1
        consistent = sum(sign == majority_sign for sign in nonzero)
        stable.append(
            {
                "feature": feature,
                "splits_with_effect": len(clean),
                "consistent_splits": int(consistent),
                "majority_direction": "higher_in_winners" if majority_sign > 0 else "lower_in_winners",
                "median_abs_cliffs_delta": float(
                    np.median([abs(float(item["cliffs_delta_winner_vs_loser"])) for item in clean])
                ),
                "median_cliffs_delta": float(
                    np.median([float(item["cliffs_delta_winner_vs_loser"]) for item in clean])
                ),
                "median_diff": float(np.median([float(item["median_diff_winner_minus_loser"]) for item in clean])),
            }
        )
    stable.sort(key=lambda row: (row["consistent_splits"], row["median_abs_cliffs_delta"]), reverse=True)
    return {
        "focus": {"bucket": FOCUS_BUCKET, "right": FOCUS_RIGHT},
        "by_split": by_split,
        "stable_proxy_only_winner_separators": stable,
    }


def _fmt(value: object, digits: int = 0) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Protocol 043 Post-Open Put Fragility Diagnostic",
        "",
        "No paid data. No model changes. Focus: proxy-only versus official-only post-open morning puts.",
        "",
        "## Category Summary",
        "",
        "| Split | Proxy-Only Trades | Proxy-Only PnL | Official-Only Trades | Official-Only PnL | Missed Winner PnL | Avoided Loser PnL | Added Winner PnL | Added Loser PnL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, row in payload["by_split"].items():
        lines.append(
            f"| {split} | {_fmt(row['proxy_only_focus']['trades'])} | {_fmt(row['proxy_only_focus']['pnl'])} | "
            f"{_fmt(row['official_only_focus']['trades'])} | {_fmt(row['official_only_focus']['pnl'])} | "
            f"{_fmt(row['proxy_only_missed_winners']['pnl'])} | {_fmt(row['proxy_only_avoided_losers']['pnl'])} | "
            f"{_fmt(row['official_only_added_winners']['pnl'])} | {_fmt(row['official_only_added_losers']['pnl'])} |"
        )
    lines += [
        "",
        "## Strongest Stable Separators In Proxy-Only Puts",
        "",
        "Positive direction means the feature is higher in missed winners than avoided losers.",
        "",
        "| Feature | Direction | Consistent Splits | Median Cliff Delta | Median Diff |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["stable_proxy_only_winner_separators"][:15]:
        lines.append(
            f"| {row['feature']} | {row['majority_direction']} | {row['consistent_splits']}/{row['splits_with_effect']} | "
            f"{_fmt(row['median_cliffs_delta'], 3)} | {_fmt(row['median_diff'], 3)} |"
        )
    lines += [
        "",
        "## Per-Split Feature Detail",
        "",
    ]
    for split, row in payload["by_split"].items():
        lines += [
            f"### {split}",
            "",
            "| Feature | Missed Winner Median | Avoided Loser Median | Diff | Cliff Delta |",
            "|---|---:|---:|---:|---:|",
        ]
        ranked = sorted(
            row["proxy_missed_vs_avoided_features"],
            key=lambda item: abs(float(item.get("cliffs_delta_winner_vs_loser") or 0.0)),
            reverse=True,
        )
        for item in ranked[:12]:
            lines.append(
                f"| {item['feature']} | {_fmt(item['winner_median'], 3)} | {_fmt(item['loser_median'], 3)} | "
                f"{_fmt(item['median_diff_winner_minus_loser'], 3)} | {_fmt(item['cliffs_delta_winner_vs_loser'], 3)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    proxy_rows = _load_dir(args.proxy_dir, protocol="proxy")
    official_rows = _load_dir(args.official_dir, protocol="official")
    payload = _sanitize(_build(proxy_rows, official_rows))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_markdown(args.out_dir / "report.md", payload)
    print(args.out_dir / "report.json")
    print(args.out_dir / "report.md")
    for row in payload["stable_proxy_only_winner_separators"][:8]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
