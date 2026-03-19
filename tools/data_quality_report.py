#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.prepare import FEATURE_NAMES, load_data  # noqa: E402


FEATURE_CACHE_DIR = Path.home() / ".cache" / "autoresearch-trading" / "features"
DATA_QUALITY_REPORT_JSON = FEATURE_CACHE_DIR / "data_quality_report.json"
DATA_QUALITY_FINGERPRINT_JSON = FEATURE_CACHE_DIR / "data_quality_fingerprint.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return dt.datetime.now().isoformat()


def _compute_data_quality_report(data: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    features = data["features"]
    dates = list(data.get("dates", []))
    timestamps = list(data.get("timestamps", []))
    n_bars = int(features.shape[0])
    n_features = int(features.shape[1]) if features.ndim == 2 else 0
    nan_mask = torch.isnan(features)
    overall_nan_pct = float(nan_mask.float().mean().item() * 100.0) if n_bars and n_features else 0.0

    nan_rate_by_feature = []
    if n_features:
        per_feat = nan_mask.float().mean(dim=0).cpu().numpy()
        for i in range(n_features):
            nan_rate_by_feature.append(
                {
                    "index": i,
                    "name": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                    "nan_rate": float(per_feat[i]),
                }
            )

    # Bar-of-day missingness profile (0..389)
    by_minute_sum = [0.0] * 390
    by_minute_count = [0] * 390
    row_nan = nan_mask.float().mean(dim=1).cpu().numpy() if n_bars and n_features else []
    minute_idx = 0
    prev_date = None
    for i, d in enumerate(dates):
        if d != prev_date:
            minute_idx = 0
            prev_date = d
        else:
            minute_idx += 1
        bod = max(0, min(389, minute_idx))
        by_minute_sum[bod] += float(row_nan[i]) if i < len(row_nan) else 0.0
        by_minute_count[bod] += 1
    tod_missingness = []
    for m in range(390):
        if by_minute_count[m] == 0:
            continue
        tod_missingness.append(
            {
                "bar_of_day": m,
                "avg_nan_rate": by_minute_sum[m] / by_minute_count[m],
                "samples": by_minute_count[m],
            }
        )
    tod_missingness = sorted(tod_missingness, key=lambda x: x["avg_nan_rate"], reverse=True)[:25]

    def _coverage_for(key: str) -> float | None:
        arr = data.get(key)
        if arr is None:
            return None
        return float((~torch.isnan(arr)).float().mean().item())

    option_coverage = {
        "atm_call_prices": _coverage_for("atm_call_prices"),
        "atm_put_prices": _coverage_for("atm_put_prices"),
        "otm5_call_prices": _coverage_for("otm5_call_prices"),
        "otm5_put_prices": _coverage_for("otm5_put_prices"),
        "otm10_call_prices": _coverage_for("otm10_call_prices"),
        "otm10_put_prices": _coverage_for("otm10_put_prices"),
        "otm15_call_prices": _coverage_for("otm15_call_prices"),
        "otm15_put_prices": _coverage_for("otm15_put_prices"),
        "otm20_call_prices": _coverage_for("otm20_call_prices"),
        "otm20_put_prices": _coverage_for("otm20_put_prices"),
    }
    sidecar_coverage = {
        "action_spread_bps": _coverage_for("action_spread_bps"),
        "action_quote_age_s": _coverage_for("action_quote_age_s"),
        "action_size": _coverage_for("action_size"),
        "action_quality_score": _coverage_for("action_quality_score"),
        "action_slippage_bps": _coverage_for("action_slippage_bps"),
        "action_cost_bps": _coverage_for("action_cost_bps"),
        "actionable_mask": _coverage_for("actionable_mask"),
        "risk_state_mask": _coverage_for("risk_state_mask"),
        "supervision_weight": _coverage_for("supervision_weight"),
    }

    date_start = min(dates) if dates else None
    date_end = max(dates) if dates else None
    num_days = len(set(dates)) if dates else 0
    summary = {
        "n_bars": n_bars,
        "n_features": n_features,
        "date_start": date_start,
        "date_end": date_end,
        "num_days": num_days,
        "overall_nan_pct": round(overall_nan_pct, 6),
        "option_coverage": {
            k: (None if v is None else round(v, 6))
            for k, v in option_coverage.items()
        },
        "sidecar_coverage": {
            k: (None if v is None else round(v, 6))
            for k, v in sidecar_coverage.items()
        },
    }
    fingerprint = _sha256_text(json.dumps(summary, sort_keys=True))

    return {
        "created_at": _now_iso(),
        "fingerprint": fingerprint,
        "summary": summary,
        "nan_rate_by_feature": nan_rate_by_feature,
        "time_of_day_missingness_top": tod_missingness,
        "timestamp_samples": {
            "first": str(timestamps[0]) if timestamps else None,
            "last": str(timestamps[-1]) if timestamps else None,
        },
    }


def _load_cached_fingerprint() -> dict[str, Any] | None:
    if not DATA_QUALITY_FINGERPRINT_JSON.exists():
        return None
    try:
        return json.loads(DATA_QUALITY_FINGERPRINT_JSON.read_text())
    except Exception:
        return None


def _save_report_and_fingerprint(report: dict[str, Any], output: Path) -> None:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    DATA_QUALITY_REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
    DATA_QUALITY_FINGERPRINT_JSON.write_text(
        json.dumps(
            {
                "fingerprint": report.get("fingerprint"),
                "summary": report.get("summary", {}),
                "created_at": report.get("created_at"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate data.pt quality report + fingerprint")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_QUALITY_REPORT_JSON),
        help="Report output path (default: cache report path)",
    )
    parser.add_argument(
        "--allow-fingerprint-change",
        action="store_true",
        help="Accept and persist a changed fingerprint",
    )
    parser.add_argument(
        "--no-check-fingerprint",
        action="store_true",
        help="Skip changed-fingerprint gate (always persist)",
    )
    args = parser.parse_args()

    data = load_data()
    report = _compute_data_quality_report(data, list(FEATURE_NAMES))
    output = Path(args.output).expanduser().resolve()
    cached = _load_cached_fingerprint()
    current_fp = report.get("fingerprint")
    cached_fp = cached.get("fingerprint") if cached else None

    print("=== DATA QUALITY REPORT ===")
    print(f"fingerprint: {current_fp}")
    print(
        "summary: "
        f"{report['summary']['n_bars']} bars | "
        f"{report['summary']['n_features']} features | "
        f"{report['summary']['date_start']} -> {report['summary']['date_end']} | "
        f"NaN={report['summary']['overall_nan_pct']:.2f}%"
    )

    changed = bool(cached_fp and cached_fp != current_fp)
    if changed and not args.no_check_fingerprint and not args.allow_fingerprint_change:
        print("ERROR: fingerprint changed vs cached baseline.")
        print(f"  cached:  {cached_fp}")
        print(f"  current: {current_fp}")
        print("  Re-run with --allow-fingerprint-change to accept this new baseline.")
        return 2

    if changed and args.allow_fingerprint_change:
        print("WARN: fingerprint changed and override is enabled; updating cached baseline.")

    _save_report_and_fingerprint(report, output)
    print(f"saved report: {output}")
    print(f"saved cache report: {DATA_QUALITY_REPORT_JSON}")
    print(f"saved cache fingerprint: {DATA_QUALITY_FINGERPRINT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
