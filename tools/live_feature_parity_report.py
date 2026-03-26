#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.prepare import FEATURE_NAMES  # noqa: E402


def _source_bucket(feature_idx: int) -> str:
    # Aligned to 32-feature FEATURE_NAMES in prepare.py
    if feature_idx < 2:
        return "price_returns"        # ret_6, ret_12
    if feature_idx < 5:
        return "volume"               # volume_ratio, volume_zscore, volume_at_price_pctile
    if feature_idx < 8:
        return "volatility"           # bar_range, realized_vol, range_ratio
    if feature_idx < 10:
        return "vwap"                 # vwap_dist, vwap_slope
    if feature_idx < 12:
        return "session_structure"    # ib_width, session_range_pct
    if feature_idx < 14:
        return "key_levels"           # prev_high_dist, prev_low_dist
    if feature_idx < 17:
        return "trend"                # ema_cross, consec_direction, speed_estimate
    if feature_idx < 19:
        return "microstructure"       # gap, inside_bar
    if feature_idx < 22:
        return "time"                 # minutes_to_close, time_sin, time_cos
    if feature_idx < 24:
        return "options"              # atm_iv, iv_skew
    if feature_idx < 26:
        return "vix_regime"           # vix_regime, vrp
    if feature_idx < 29:
        return "greeks"               # atm_gamma, atm_theta_per_bar, charm_estimate
    if feature_idx < 30:
        return "bollinger"            # bollinger_position
    return "range_extras"             # rsi_14, session_range_position


def _status(pct: float) -> str:
    if pct >= 0.98:
        return "good"
    if pct >= 0.85:
        return "degraded"
    return "missing"


def _coerce_mask(payload: dict[str, Any], n: int) -> list[int] | None:
    mask = payload.get("non_nan_mask")
    if isinstance(mask, list):
        if len(mask) >= n:
            return [1 if int(x) else 0 for x in mask[:n]]
        return None

    missing_idx = payload.get("missing_feature_indices")
    if isinstance(missing_idx, list):
        out = [1] * n
        for idx in missing_idx:
            try:
                i = int(idx)
                if 0 <= i < n:
                    out[i] = 0
            except Exception:
                continue
        return out
    return None


def build_report(audit_path: str) -> dict[str, Any]:
    n = len(FEATURE_NAMES)
    present_counts = [0] * n
    total = 0
    bad_rows = 0

    with open(audit_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            if row.get("event") != "bar_snapshot":
                continue
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                bad_rows += 1
                continue
            mask = _coerce_mask(payload, n)
            if mask is None:
                bad_rows += 1
                continue
            total += 1
            for i in range(n):
                present_counts[i] += int(mask[i])

    if total == 0:
        raise RuntimeError(
            "No usable bar_snapshot rows with per-feature masks. "
            "Run paper_live after this telemetry update to generate parity evidence."
        )

    features: list[dict[str, Any]] = []
    bucket_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"present": 0.0, "total": 0.0})
    for idx, name in enumerate(FEATURE_NAMES):
        pct = present_counts[idx] / total
        bucket = _source_bucket(idx)
        bucket_stats[bucket]["present"] += present_counts[idx]
        bucket_stats[bucket]["total"] += total
        features.append(
            {
                "index": idx,
                "name": name,
                "source_bucket": bucket,
                "present_bars": int(present_counts[idx]),
                "total_bars": int(total),
                "present_pct": float(pct),
                "status": _status(pct),
            }
        )

    bucket_summary = []
    for bucket, acc in sorted(bucket_stats.items()):
        pct = acc["present"] / max(acc["total"], 1.0)
        bucket_summary.append(
            {
                "source_bucket": bucket,
                "present_pct": float(pct),
                "status": _status(pct),
            }
        )

    return {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "schema_version": "feature_parity_v1",
        "audit_path": os.path.abspath(audit_path),
        "bar_snapshots": int(total),
        "bad_rows": int(bad_rows),
        "features": features,
        "bucket_summary": bucket_summary,
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Live Feature Parity Report")
    lines.append("")
    lines.append(f"- generated_at: `{report['generated_at']}`")
    lines.append(f"- audit_path: `{report['audit_path']}`")
    lines.append(f"- bar_snapshots: `{report['bar_snapshots']}`")
    lines.append(f"- bad_rows: `{report['bad_rows']}`")
    lines.append("")
    lines.append("## Source Buckets")
    lines.append("")
    lines.append("| bucket | present_pct | status |")
    lines.append("|---|---:|---|")
    for row in report["bucket_summary"]:
        lines.append(f"| {row['source_bucket']} | {row['present_pct']:.2%} | {row['status']} |")
    lines.append("")
    lines.append(f"## Features ({len(report['features'])})")
    lines.append("")
    lines.append("| idx | feature | source | present_pct | status |")
    lines.append("|---:|---|---|---:|---|")
    for row in report["features"]:
        lines.append(
            f"| {row['index']} | {row['name']} | {row['source_bucket']} | {row['present_pct']:.2%} | {row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize live IBKR per-feature availability.")
    parser.add_argument("--audit-path", default=os.path.join("results", "live", "audit.jsonl"))
    parser.add_argument("--out-json", default=os.path.join("results", "live", "feature_parity_report.json"))
    parser.add_argument("--out-md", default=os.path.join("results", "live", "feature_parity_report.md"))
    args = parser.parse_args()

    report = build_report(args.audit_path)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(args.out_md, "w") as f:
        f.write(_to_markdown(report))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

